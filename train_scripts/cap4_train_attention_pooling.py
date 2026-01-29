import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy

# ==============================================================================
# 1. CONFIGURAÇÕES E CAMINHOS
# ==============================================================================
BASE_PATH = 'experiments_cap_4'
NLI_CSV_PATH = '/home/yansoares/research_embeddings_fine_tuning/data/nli_optimized_for_layer_search_full.csv'
SENTEVAL_DATA_PATH = '/home/yansoares/research_embeddings_fine_tuning/data'

# Pesos de Camada (Vencedor do Cap 3)
FUSION_WEIGHTS_PATH = os.path.join(BASE_PATH, 'dynamic_weights_universal_v5.json')

# Onde salvar o modelo final
SAVE_DIR = os.path.join(BASE_PATH, 'results_cap4_final')
os.makedirs(SAVE_DIR, exist_ok=True)

CONFIG = {
    'model_name': 'microsoft/deberta-v3-base',
    'batch_size': 512,
    'epochs': 10,             # Atenção converge muito rápido no NLI
    'lr': 1e-3,              # Learning Rate mais alto para a camada de atenção
    'max_len': 128,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # AQUI ESTÁ A CHAVE: Usamos o vetor UNIVERSAL do Cap 3
    'fusion_profile': 'AVG_SENTEVAL' 
}

print(f"🚀 Iniciando Treino Universal no dispositivo: {CONFIG['device']}")
print(f"📂 Perfil de Camadas: {CONFIG['fusion_profile']}")

# ==============================================================================
# 2. ARQUITETURA DO MODELO UNIVERSAL
# ==============================================================================
class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention_projection = nn.Linear(in_dim, in_dim)
        self.context_vector = nn.Linear(in_dim, 1, bias=False)

    def forward(self, hidden_states, mask):
        # 1. Calcula score de atenção para cada token
        weights = torch.tanh(self.attention_projection(hidden_states))
        weights = self.context_vector(weights).squeeze(-1)
        
        # 2. Aplica máscara (ignora padding)
        weights = weights.masked_fill(mask == 0, -1e9)
        
        # 3. Softmax para obter probabilidades
        weights = F.softmax(weights, dim=1)
        
        # 4. Soma ponderada dos vetores
        weights_expanded = weights.unsqueeze(-1)
        return torch.sum(hidden_states * weights_expanded, dim=1)

class UniversalModel(nn.Module):
    def __init__(self, model_name, fusion_path, profile_key):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # --- PARTE 1: FUSÃO DE CAMADAS (CONGELADA) ---
        self.layer_weights = nn.Parameter(torch.zeros(self.config.num_hidden_layers), requires_grad=False)
        self._load_layer_weights(fusion_path, profile_key)
        
        # --- PARTE 2: ATENÇÃO (TREINÁVEL) ---
        self.att_pool = AttentionPooling(self.config.hidden_size)
        
        # --- PARTE 3: CLASSIFICADOR NLI (TREINÁVEL) ---
        # Usado apenas para guiar o aprendizado da atenção
        self.classifier = nn.Linear(self.config.hidden_size, 3) 
        self.dropout = nn.Dropout(0.1)

        # Congela o Backbone (DeBERTa) para focar o treino na Atenção
        for param in self.backbone.parameters(): 
            param.requires_grad = False

    def _load_layer_weights(self, path, key):
        print(f"--- Carregando Vetor de Camadas: {key} ---")
        with open(path) as f:
            data = json.load(f)
        
        if key not in data:
            raise ValueError(f"❌ Erro Crítico: Chave '{key}' não encontrada no JSON de pesos!")
        
        w = data[key]
        self.layer_weights.data = torch.tensor(w)
        print(f"✅ Pesos carregados e congelados: {w}")

    def forward(self, input_ids, attention_mask):
        # 1. Extrai Hidden States (Backbone Congelado)
        with torch.no_grad():
            outputs = self.backbone(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # 2. Aplica Pesos Universais (AVG_SENTEVAL)
        # Pega as últimas N camadas
        stacked = torch.stack(outputs.hidden_states[-len(self.layer_weights):], dim=1)
        w = self.layer_weights.view(1, -1, 1, 1).to(input_ids.device)
        fused_sequence = (stacked * w).sum(dim=1)
        
        # 3. Aplica Atenção (O que está sendo treinado)
        pooled_output = self.att_pool(fused_sequence, attention_mask)
        
        # 4. Classificação
        logits = self.classifier(self.dropout(pooled_output))
        return logits

# ==============================================================================
# 3. PREPARAÇÃO DOS DADOS
# ==============================================================================
class NLIDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Tokenização otimizada
        enc = self.tokenizer(
            str(row['premise']), 
            str(row['hypothesis']), 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True, 
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].flatten(), 
            'attention_mask': enc['attention_mask'].flatten(), 
            'labels': torch.tensor(int(row['label']))
        }

# ==============================================================================
# 4. LOOP DE TREINAMENTO
# ==============================================================================
def train():
    print("\n--- Preparando Dados NLI ---")
    if not os.path.exists(NLI_CSV_PATH):
        raise FileNotFoundError(f"CSV não encontrado: {NLI_CSV_PATH}")
        
    df = pd.read_csv(NLI_CSV_PATH)
    # Divisão Stratified para manter balanceamento das classes
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
    print(f"Treino: {len(train_df)} | Validação: {len(val_df)}")
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    train_loader = DataLoader(NLIDataset(train_df, tokenizer, CONFIG['max_len']), batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(NLIDataset(val_df, tokenizer, CONFIG['max_len']), batch_size=CONFIG['batch_size'])

    # Instancia Modelo Universal
    model = UniversalModel(CONFIG['model_name'], FUSION_WEIGHTS_PATH, CONFIG['fusion_profile']).to(CONFIG['device'])
    
    # Otimizador (apenas parâmetros com requires_grad=True serão atualizados)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_att_weights = None
    patience_counter = 0

    print("\n>>> INICIANDO TREINO (Foco: Atenção) <<<")
    
    for epoch in range(CONFIG['epochs']):
        # Treino
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch['input_ids'].to(CONFIG['device']), batch['attention_mask'].to(CONFIG['device']))
            loss = criterion(out, batch['labels'].to(CONFIG['device']))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validação
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch['input_ids'].to(CONFIG['device']), batch['attention_mask'].to(CONFIG['device']))
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                targets.extend(batch['labels'].cpu().numpy())
        
        val_acc = accuracy_score(targets, preds)
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Copia os pesos da camada de atenção (nosso objetivo final)
            best_att_weights = copy.deepcopy(model.att_pool.state_dict())
            patience_counter = 0
            print(" -> 🌟 Novo Melhor Modelo! Pesos capturados.")
        else:
            patience_counter += 1
            if patience_counter >= 2: # Early stopping agressivo
                print(" -> Early Stopping ativado.")
                break
    
    return best_att_weights, best_val_acc

# ==============================================================================
# 5. SALVAMENTO E FINALIZAÇÃO
# ==============================================================================
if __name__ == "__main__":
    trained_weights, final_acc = train()
    
    if trained_weights:
        save_path = os.path.join(SAVE_DIR, "universal_attention_weights_final.pt")
        
        # O TRUQUE DO "UNIVERSAL":
        # Salvamos o mesmo peso para TODAS as chaves possíveis.
        # Assim, quando o SentEval rodar na task 'MR', ele vai buscar a chave 'MR' 
        # e encontrar este peso treinado no NLI+AVG_SENTEVAL.
        final_dict = {
            'UNIVERSAL': trained_weights,
            'AVG_SENTEVAL': trained_weights,
            'NLI': trained_weights,
            # Mapeamento para todas as tasks do SentEval (Classificação)
            'MR': trained_weights, 'CR': trained_weights, 'SUBJ': trained_weights, 'MPQA': trained_weights, 
            'SST2': trained_weights, 'TREC': trained_weights, 'MRPC': trained_weights,
            # Mapeamento para STS (Similaridade)
            'STSBenchmark': trained_weights, 'SICKRelatedness': trained_weights, 
            'STS12': trained_weights, 'STS13': trained_weights, 'STS14': trained_weights, 'STS15': trained_weights, 'STS16': trained_weights
        }
        
        torch.save(final_dict, save_path)
        
        print(f"\n✅ TREINO FINALIZADO COM SUCESSO!")
        print(f"Acurácia no NLI (Validation): {final_acc:.4f}")
        print(f"Arquivo salvo em: {save_path}")
        print("-" * 60)
        print("PRÓXIMO PASSO (AVALIAÇÃO FINAL):")
        print("Rode o script 'main.py' com os seguintes argumentos:")
        print(f"  --dynamic_weights_path {FUSION_WEIGHTS_PATH}")
        print(f"  --attention_weights_path {save_path}")
        print(f"  --pooling ATTENTION")
        print("-" * 60)
    else:
        print("❌ Falha no treinamento.")