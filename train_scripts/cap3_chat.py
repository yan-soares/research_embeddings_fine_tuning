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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import warnings
from tqdm.auto import tqdm
from datasets import load_dataset
from scipy.stats import spearmanr, pearsonr

# Ignora avisos
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

BASE_PATH = 'cap3_results_sts_chat'

MODELS_TO_TEST = [
    'microsoft/deberta-v3-base',
    #'answerdotai/ModernBERT-base',
    #'microsoft/deberta-v3-large',
    #'google-bert/bert-base-uncased',
    #'FacebookAI/roberta-base',
]

SEEDS = [42]#, 0, 1234, 2025, 999]

# ==============================================================================
# 1. HIPERPARÂMETROS GERAIS (AGORA SÓ STS)
# ==============================================================================
MAX_EPOCHS = 50
PATIENCE = 3

CONFIG = {
    'epochs': MAX_EPOCHS,
    'lr': 1e-3,
    'max_len': 128,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'task': 'STS',  # <-- só STS
}

if CONFIG['device'] == 'cpu':
    print("⚠️⚠️⚠️ ATENÇÃO: VOCÊ ESTÁ RODANDO EM CPU! VAI DEMORAR MUITO! ⚠️⚠️⚠️")
else:
    print(f"🚀 Rodando no dispositivo: {CONFIG['device']}")
    print(f"   Placa detectada: {torch.cuda.get_device_name(0)}")

# ==============================================================================
# 2. MODELO
# ==============================================================================
class DynamicFusionLayer(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, all_hidden_states):
        stacked_layers = torch.stack(all_hidden_states[1:], dim=1)  # [B, L, T, H]
        norm_weights = F.softmax(self.layer_weights, dim=0)          # [L]
        weights_reshaped = norm_weights.view(1, -1, 1, 1)
        fused_embedding = (stacked_layers * weights_reshaped).sum(dim=1)  # [B, T, H]
        return fused_embedding, norm_weights


class STSModel(nn.Module):
    """
    Regressão: prevê um escalar de similaridade.
    Backbone congelado, só treina a fusão + cabeça linear.
    """
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)

        if hasattr(self.config, "num_hidden_layers"):
            self.n_layers = self.config.num_hidden_layers
        elif hasattr(self.config, "n_layers"):
            self.n_layers = self.config.n_layers
        else:
            self.n_layers = 22
            print(f"Aviso: Não foi possível detectar n_layers da config. Usando fallback {self.n_layers}")

        print(f"Modelo {model_name} inicializado com {self.n_layers} camadas para fusão.")

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.fusion = DynamicFusionLayer(num_layers=self.n_layers)
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.backbone(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        fused_sequence, weights = self.fusion(outputs.hidden_states)

        # mean pooling mask-aware
        mask_expanded = attention_mask.unsqueeze(-1).expand(fused_sequence.size()).float()
        sum_embeddings = torch.sum(fused_sequence * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-4)
        pooled = sum_embeddings / sum_mask

        score = self.regressor(self.dropout(pooled)).squeeze(-1)  # [B]
        return score, weights


# ==============================================================================
# 3. DATASET
# ==============================================================================
class STSDataset(Dataset):
    def __init__(self, pairs, labels, tokenizer, max_len):
        self.pairs = pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): 
        return len(self.pairs)

    def __getitem__(self, idx):
        s1, s2 = self.pairs[idx]
        enc = self.tokenizer(
            str(s1), str(s2),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }


# ==============================================================================
# 4. UTILS
# ==============================================================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def clean_for_json(obj):
    if isinstance(obj, (np.integer, int)): return int(obj)
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, list): return [clean_for_json(i) for i in obj]
    return obj

def load_data_stsb(normalize=True):
    ds = load_dataset("glue", "stsb")
    X_train = list(zip(ds["train"]["sentence1"], ds["train"]["sentence2"]))
    y_train = np.array(ds["train"]["label"], dtype=np.float32)

    X_val = list(zip(ds["validation"]["sentence1"], ds["validation"]["sentence2"]))
    y_val = np.array(ds["validation"]["label"], dtype=np.float32)

    # STS-B labels são 0..5. Normalizar para 0..1 ajuda na estabilidade.
    if normalize:
        y_train = y_train / 5.0
        y_val = y_val / 5.0

    return X_train, X_val, y_train, y_val


# ==============================================================================
# 5. ENGINE DE TREINO (AGORA SÓ STS)
# ==============================================================================
def train_sts_engine(seed_idx=0, total_seeds=5):
    X_train, X_val, y_train, y_val = load_data_stsb(normalize=True)

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

    train_ds = STSDataset(X_train, y_train, tokenizer, CONFIG['max_len'])
    val_ds   = STSDataset(X_val,   y_val,   tokenizer, CONFIG['max_len'])

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG['batch_size'], num_workers=4, pin_memory=True)

    model = STSModel(CONFIG['model_name']).to(CONFIG['device'])

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.MSELoss()

    history = {
        'train_loss': [],
        'val_spearman': [],
        'val_pearson': [],
        'weight_evolution': []
    }

    best_spearman = -1e9
    best_weights = None
    patience_counter = 0

    epochs_bar = tqdm(range(CONFIG['epochs']), desc=f"Seed {seed_idx+1}/{total_seeds} | STS", leave=False)

    for epoch in epochs_bar:
        # Train
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            preds, _ = model(
                batch['input_ids'].to(CONFIG['device']),
                batch['attention_mask'].to(CONFIG['device'])
            )
            loss = criterion(preds, batch['labels'].to(CONFIG['device']).float())
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        # Val
        model.eval()
        preds_all, targets_all = [], []
        curr_w = None

        with torch.no_grad():
            for batch in val_loader:
                preds, w = model(
                    batch['input_ids'].to(CONFIG['device']),
                    batch['attention_mask'].to(CONFIG['device'])
                )
                preds_all.extend(preds.detach().cpu().numpy().tolist())
                targets_all.extend(batch['labels'].cpu().numpy().tolist())

                if curr_w is None:
                    curr_w = w.detach().cpu().numpy().tolist()

        avg_loss = total_loss / max(1, len(train_loader))

        sp = float(spearmanr(targets_all, preds_all).correlation)
        pr = float(pearsonr(targets_all, preds_all)[0])

        # NaN safety (pode acontecer no começo)
        if np.isnan(sp): sp = -1.0
        if np.isnan(pr): pr = -1.0

        history['train_loss'].append(avg_loss)
        history['val_spearman'].append(sp)
        history['val_pearson'].append(pr)
        history['weight_evolution'].append(curr_w)

        scheduler.step(sp)
        epochs_bar.set_postfix({'Loss': f"{avg_loss:.4f}", 'Spr': f"{sp:.4f}", 'Prs': f"{pr:.4f}"})

        # Early stopping pelo Spearman
        if sp > best_spearman:
            best_spearman = sp
            best_weights = curr_w
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                epochs_bar.write(f"   -> Early Stopping na época {epoch+1}")
                break

    return best_weights, best_spearman, history


# ==============================================================================
# 6. MAIN (LOOP DE MODELOS E SEEDS) - AGORA SÓ STS
# ==============================================================================
if __name__ == "__main__":

    for model_name in MODELS_TO_TEST:
        print(f"\n\n{'█'*60}")
        print(f"🤖 PREPARANDO MODELO: {model_name}")
        print(f"{'█'*60}")

        if 'large' in model_name.lower():
            CONFIG['batch_size'] = 128
        else:
            CONFIG['batch_size'] = 256

        CONFIG['model_name'] = model_name

        print(f"⏳ Verificando e carregando {model_name} na memória...")
        try:
            AutoConfig.from_pretrained(model_name)
            print(f"✅ Modelo verificado! Iniciando experimentos STS...")
        except Exception as e:
            print(f"❌ Erro ao carregar {model_name}. Pulando... {e}")
            continue

        safe_name = model_name.replace('/', '_')
        MODEL_SAVE_DIR = os.path.join(BASE_PATH, f'results_cap3_STS_{safe_name}')
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

        sts_weights_runs = []
        sts_score_runs = []
        all_histories_dump = {}

        for i, seed in enumerate(SEEDS):
            set_seed(seed)
            weights, spearman_best, history = train_sts_engine(seed_idx=i, total_seeds=len(SEEDS))
            if weights is not None:
                sts_weights_runs.append(weights)
                sts_score_runs.append(spearman_best)
                all_histories_dump[f"STS_seed_{seed}"] = history
            else:
                print(f"   ❌ Falha na seed {seed}")

        if len(sts_weights_runs) > 0:
            weights_np = np.array(sts_weights_runs)  # [runs, layers]
            scores_np = np.array(sts_score_runs)

            mean_weights = np.mean(weights_np, axis=0)
            std_weights = np.std(weights_np, axis=0)

            mean_sp = float(np.mean(scores_np))
            std_sp  = float(np.std(scores_np))

            print(f"\n   📊 STS (Spearman): {mean_sp:.4f} ± {std_sp:.4f}")

            # Salva score
            df_score = pd.DataFrame([{
                'Task': 'STS',
                'Score': f"{mean_sp:.4f} ± {std_sp:.4f}"
            }])
            df_score.to_csv(os.path.join(MODEL_SAVE_DIR, 'final_scores.csv'), index=False)

            # Salva pesos médios
            df_weights = pd.DataFrame([mean_weights], index=['STS'])
            df_weights.columns = [f"L{i+1}" for i in range(df_weights.shape[1])]
            df_weights.loc['AVG_ALL'] = df_weights.mean(axis=0)

            with open(os.path.join(MODEL_SAVE_DIR, 'dynamic_weights_mean.json'), 'w') as f:
                json.dump(df_weights.T.to_dict(orient='list'), f, indent=4)

            # Salva desvio padrão
            df_std = pd.DataFrame([std_weights], index=['STS'])
            df_std.columns = [f"L{i+1}" for i in range(df_std.shape[1])]
            df_std.to_csv(os.path.join(MODEL_SAVE_DIR, 'dynamic_weights_std.csv'))

            # Históricos
            hist_cleaned = {k: {m: clean_for_json(v) for m, v in h.items()} for k, h in all_histories_dump.items()}
            with open(os.path.join(MODEL_SAVE_DIR, 'all_histories.json'), 'w') as f:
                json.dump(hist_cleaned, f)

            # Heatmap dos pesos médios (STS + AVG_ALL)
            plt.figure(figsize=(18, 6))
            sns.heatmap(df_weights, annot=True, fmt=".3f", cmap="YlGnBu")
            plt.title(f"Pesos Médios das Camadas - STS - {model_name} (5 Seeds)")
            plt.tight_layout()
            plt.savefig(os.path.join(MODEL_SAVE_DIR, "heatmap_mean_weights.png"), dpi=300)
            plt.close()

            # Heatmap do desvio padrão
            plt.figure(figsize=(18, 4))
            sns.heatmap(df_std, annot=True, fmt=".3f", cmap="Reds", annot_kws={"size": 10})
            plt.title(f"DESVIO PADRÃO dos Pesos - STS - {model_name}", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(MODEL_SAVE_DIR, "heatmap_std_weights.png"), dpi=300)
            plt.close()

            print(f"✅ Resultados salvos em: {MODEL_SAVE_DIR}")

    print("\n🏁 EXECUÇÃO COMPLETA DE TODOS OS MODELOS (STS).")
