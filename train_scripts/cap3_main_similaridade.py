import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
from tqdm.auto import tqdm
from scipy.stats import pearsonr

# --- AJUSTE DE CAMINHOS ---
PROJECT_ROOT = '/home/yansoares/research_embeddings_fine_tuning'
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

BASE_PATH = os.path.join(PROJECT_ROOT, 'train_scripts/cap3_results_sts_best')
STS_DATA_PATH = os.path.join(PROJECT_ROOT, 'data/downstream/stsbenchmark') 

MODELS_TO_TEST = ['microsoft/deberta-v3-base'] # Adicione outros aqui
SEEDS = [42] 
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. ARQUITETURA (FUSÃO + POOLING AVG)
# ==============================================================================
class DynamicFusionLayer(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, all_hidden_states):
        stacked_layers = torch.stack(all_hidden_states[1:], dim=1)
        norm_weights = F.softmax(self.layer_weights, dim=0)
        weights_reshaped = norm_weights.view(1, -1, 1, 1)
        fused_embedding = (stacked_layers * weights_reshaped).sum(dim=1)
        return fused_embedding, norm_weights

class STSFusionModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.config = self.backbone.config
        self.n_layers = getattr(self.config, "num_hidden_layers", 12)
        for param in self.backbone.parameters(): param.requires_grad = False
        self.fusion = DynamicFusionLayer(num_layers=self.n_layers)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.backbone(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        fused_sequence, weights = self.fusion(outputs.hidden_states)
        
        # Mean Pooling (AVG)
        mask = attention_mask.unsqueeze(-1).expand(fused_sequence.size()).float()
        pooled = torch.sum(fused_sequence * mask, 1) / torch.clamp(mask.sum(1), min=1e-4)
        return pooled, weights

# ==============================================================================
# 2. DATASET E LOADER
# ==============================================================================
class STSDataset(Dataset):
    def __init__(self, s1, s2, scores, tokenizer, max_len):
        self.s1, self.s2, self.scores = s1, s2, scores
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self): return len(self.s1)

    def __getitem__(self, idx):
        def t(text): return self.tokenizer(str(text), max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        enc_a, enc_b = t(self.s1[idx]), t(self.s2[idx])
        return {
            'a_ids': enc_a['input_ids'].flatten(), 'a_mask': enc_a['attention_mask'].flatten(),
            'b_ids': enc_b['input_ids'].flatten(), 'b_mask': enc_b['attention_mask'].flatten(),
            'label': torch.tensor(self.scores[idx], dtype=torch.float)
        }

def load_sts_data(filename):
    path = os.path.join(STS_DATA_PATH, filename)
    df = pd.read_csv(path, sep='\t', header=None, quoting=3, on_bad_lines='skip', 
                     usecols=[4, 5, 6], names=['score', 's1', 's2'])
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df = df.dropna(subset=['s1', 's2', 'score'])
    if df['score'].max() > 1.0: df['score'] = df['score'] / 5.0
    return df

# ==============================================================================
# 3. ENGINE DE TREINO COM VALIDAÇÃO (PEARSON)
# ==============================================================================
def train_sts_engine(model_name, device, seed):
    torch.manual_seed(seed)
    train_df = load_sts_data('sts-train.csv')
    val_df = load_sts_data('sts-dev.csv') # Importante ter o dev para validar
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = STSFusionModel(model_name).to(device)
    optimizer = optim.AdamW(model.fusion.parameters(), lr=2e-3)
    criterion = nn.MSELoss()

    train_loader = DataLoader(STSDataset(train_df['s1'].values, train_df['s2'].values, train_df['score'].values, tokenizer, 128), 
                              batch_size=256, shuffle=True)
    val_loader = DataLoader(STSDataset(val_df['s1'].values, val_df['s2'].values, val_df['score'].values, tokenizer, 128), 
                            batch_size=256)

    best_weights = None
    best_pearson = -1.0
    
    epochs_bar = tqdm(range(15), desc=f"Training {model_name}")

    for epoch in epochs_bar:
        # Treino
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            v_a, _ = model(batch['a_ids'].to(device), batch['a_mask'].to(device))
            v_b, _ = model(batch['b_ids'].to(device), batch['b_mask'].to(device))
            loss = criterion(F.cosine_similarity(v_a, v_b), batch['label'].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validação
        model.eval()
        all_preds, all_labels = [], []
        curr_epoch_weights = None
        
        with torch.no_grad():
            for batch in val_loader:
                v_a, w = model(batch['a_ids'].to(device), batch['a_mask'].to(device))
                v_b, _ = model(batch['b_ids'].to(device), batch['b_mask'].to(device))
                all_preds.extend(F.cosine_similarity(v_a, v_b).cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
                if curr_epoch_weights is None:
                    curr_epoch_weights = w.detach().cpu().numpy().tolist()

        # Cálculo da métrica científica
        current_pearson, _ = pearsonr(all_preds, all_labels)
        avg_loss = total_loss / len(train_loader)
        
        # Update TQDM
        epochs_bar.set_postfix({'Loss': f"{avg_loss:.4f}", 'Pearson': f"{current_pearson:.4f}"})

        # Salva se for o melhor desempenho
        if current_pearson > best_pearson:
            best_pearson = current_pearson
            best_weights = curr_epoch_weights
            # Opcional: print informativo se houver melhora significativa
            # epochs_bar.write(f"🌟 New Best Pearson: {best_pearson:.4f}")

    return best_weights, best_pearson

# ==============================================================================
# 4. EXECUÇÃO
# ==============================================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Device: {device} | Placa: {torch.cuda.get_device_name(0)}")

    for model_name in MODELS_TO_TEST:
        safe_name = model_name.replace('/', '_')
        SAVE_DIR = os.path.join(BASE_PATH, f'results_{safe_name}')
        os.makedirs(SAVE_DIR, exist_ok=True)

        print(f"\nBusca de Pesos Ótimos para: {model_name}")
        weights, score = train_sts_engine(model_name, device, SEEDS[0])
        
        # Salva o JSON final
        output_data = {"STSBenchmark": weights, "best_pearson": score}
        with open(os.path.join(SAVE_DIR, 'dynamic_weights_sts_best.json'), 'w') as f:
            json.dump(output_data, f, indent=4)
            
        print(f"✅ Concluído! Melhor Pearson: {score:.4f}")