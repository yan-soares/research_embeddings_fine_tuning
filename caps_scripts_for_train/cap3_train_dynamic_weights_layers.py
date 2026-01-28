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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import json
import warnings

# Ignora avisos
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# 0. CONFIGURAÇÃO DE CAMINHOS E LISTAS
# ==============================================================================
BASE_PATH = '/content/drive/MyDrive/Doutorado/experiments_cap3_senteval'
NLI_CSV_PATH = os.path.join(BASE_PATH, 'data', 'nli_optimized_for_layer_search.csv')
SENTEVAL_DATA_PATH = os.path.join(BASE_PATH, 'data')

# Importação condicional do SentEval (caso não esteja instalado via pip)
# Certifique-se que a pasta senteval está no path ou instalada
try:
    from senteval.binary import CREval, MREval, MPQAEval, SUBJEval
    from senteval.trec import TRECEval
    from senteval.sst import SSTEval
    from senteval.mrpc import MRPCEval
    SENTEVAL_AVAILABLE = True
except ImportError:
    print("Biblioteca SentEval não encontrada. Verifique os caminhos.")
    SENTEVAL_AVAILABLE = False

# --- AQUI ESTÃO OS LOOPS QUE VOCÊ PEDIU ---
MODELS_TO_TEST = [
    'bert-base-uncased',
    'facebook/roberta-base',
    'microsoft/deberta-v3-base'
]

SEEDS = [42, 0, 1234, 2024, 999]

# ==============================================================================
# 1. HIPERPARÂMETROS GERAIS
# ==============================================================================
MAX_EPOCHS = 40
PATIENCE = 5

CONFIG = {
    # 'model_name': SERÁ INJETADO DINAMICAMENTE NO LOOP
    'batch_size': 64,
    'epochs': MAX_EPOCHS,
    'lr': 1e-3,
    'max_len': 128,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'tasks': ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC', 'NLI']
}

if CONFIG['device'] == 'cpu':
    print("⚠️⚠️⚠️ ATENÇÃO: VOCÊ ESTÁ RODANDO EM CPU! VAI DEMORAR MUITO! ⚠️⚠️⚠️")
else:
    print(f"🚀 Rodando no dispositivo: {CONFIG['device']}")

# ==============================================================================
# 2. MODELO
# ==============================================================================
class DynamicFusionLayer(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, all_hidden_states):
        # Pula a camada 0 (embeddings puros) e empilha as layers do encoder
        stacked_layers = torch.stack(all_hidden_states[1:], dim=1)
        norm_weights = F.softmax(self.layer_weights, dim=0)
        weights_reshaped = norm_weights.view(1, -1, 1, 1)
        fused_embedding = (stacked_layers * weights_reshaped).sum(dim=1)
        return fused_embedding, norm_weights

class TaskSpecificModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Congela Backbone
        for param in self.backbone.parameters(): 
            param.requires_grad = False

        self.fusion = DynamicFusionLayer(num_layers=self.config.num_hidden_layers)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.backbone(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        fused_sequence, weights = self.fusion(outputs.hidden_states)

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(fused_sequence.size()).float()
        sum_embeddings = torch.sum(fused_sequence * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        logits = self.classifier(self.dropout(pooled_output))
        return logits, weights

# ==============================================================================
# 3. DATASET
# ==============================================================================
class UnifiedDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len, is_pair=False):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_pair = is_pair

    def __len__(self): return len(self.sentences)

    def __getitem__(self, idx):
        if self.is_pair:
            if isinstance(self.sentences[idx], (list, tuple)):
                text_a, text_b = self.sentences[idx]
            else:
                text_a = str(self.sentences[idx])
                text_b = ""
            encoding = self.tokenizer(str(text_a), str(text_b), add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
        else:
            text_a = str(self.sentences[idx])
            encoding = self.tokenizer(str(text_a), add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')

        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'labels': torch.tensor(self.labels[idx], dtype=torch.long)}

# ==============================================================================
# 4. LOADERS & UTILS
# ==============================================================================
def set_seed(seed):
    """Garante reprodutibilidade total"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def clean_for_json(obj):
    """Converte numpy types para tipos nativos do Python para salvar JSON"""
    if isinstance(obj, (np.integer, int)): return int(obj)
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, list): return [clean_for_json(i) for i in obj]
    return obj

def load_data_nli(csv_path, seed=42):
    if not os.path.exists(csv_path): return None, None, None, None
    df = pd.read_csv(csv_path)
    X = list(zip(df['premise'].astype(str), df['hypothesis'].astype(str)))
    y = df['label'].values
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

def load_data_senteval(task_name, seed=42):
    # (Lógica idêntica à sua original, mantida para brevidade)
    if task_name == 'SST2': path = os.path.join(SENTEVAL_DATA_PATH, 'downstream', 'SST', 'binary')
    elif task_name == 'TREC': path = os.path.join(SENTEVAL_DATA_PATH, 'downstream', 'TREC')
    else: path = os.path.join(SENTEVAL_DATA_PATH, 'downstream', task_name)

    if not os.path.exists(path): raise FileNotFoundError(f"Path not found: {path}")

    if task_name == 'SST2': se = SSTEval(path, nclasses=2, seed=seed)
    elif task_name == 'TREC': se = TRECEval(path, seed=seed)
    elif task_name == 'MR': se = MREval(path, seed=seed)
    elif task_name == 'CR': se = CREval(path, seed=seed)
    elif task_name == 'MPQA': se = MPQAEval(path, seed=seed)
    elif task_name == 'SUBJ': se = SUBJEval(path, seed=seed)
    elif task_name == 'MRPC': se = MRPCEval(path, seed=seed)
    else: raise ValueError(f"Unknown task: {task_name}")

    if task_name in ['MR', 'CR', 'SUBJ', 'MPQA']:
        X = [" ".join(t) for t in se.samples]
        y = se.labels
        return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    elif task_name == 'TREC':
        return [" ".join(t) for t in se.train['X']], [" ".join(t) for t in se.test['X']], se.train['y'], se.test['y']
    elif task_name == 'SST2':
        return [" ".join(t) for t in se.sst_data['train']['X']], [" ".join(t) for t in se.sst_data['dev']['X']], se.sst_data['train']['y'], se.sst_data['dev']['y']
    elif task_name == 'MRPC':
        tr, te = se.mrpc_data['train'], se.mrpc_data['test']
        return list(zip([" ".join(s) for s in tr['X_A']], [" ".join(s) for s in tr['X_B']])), list(zip([" ".join(s) for s in te['X_A']], [" ".join(s) for s in te['X_B']])), tr['y'], te['y']

# ==============================================================================
# 5. ENGINE DE TREINO
# ==============================================================================
def train_task_engine(task_name):
    # Carregamento de Dados
    try:
        if task_name == 'NLI':
            X_train, X_val, y_train, y_val = load_data_nli(NLI_CSV_PATH)
            if X_train is None: print("Dados NLI não encontrados."); return None, 0, None
        elif SENTEVAL_AVAILABLE:
            X_train, X_val, y_train, y_val = load_data_senteval(task_name)
        else:
            return None, 0, None
    except Exception as e:
        print(f"❌ Erro dados {task_name}: {e}"); return None, 0, None

    # Setup
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    is_pair = True if task_name in ['NLI', 'MRPC'] else False
    num_classes = 6 if task_name == 'TREC' else (3 if task_name == 'NLI' else 2)

    train_ds = UnifiedDataset(X_train, y_train, tokenizer, CONFIG['max_len'], is_pair)
    val_ds = UnifiedDataset(X_val, y_val, tokenizer, CONFIG['max_len'], is_pair)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'])

    model = TaskSpecificModel(CONFIG['model_name'], num_classes=num_classes).to(CONFIG['device'])
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_acc': [], 'val_f1': [], 'weight_evolution': []}
    best_val_acc = 0.0; best_weights = None; patience_counter = 0

    print(f"   -> Iniciando Treino: {task_name} ({CONFIG['model_name']})")

    for epoch in range(CONFIG['epochs']):
        # Train
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits, _ = model(batch['input_ids'].to(CONFIG['device']), batch['attention_mask'].to(CONFIG['device']))
            loss = criterion(logits, batch['labels'].to(CONFIG['device']))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Val
        model.eval()
        preds, targets = [], []
        curr_w = None
        with torch.no_grad():
            for batch in val_loader:
                logits, w = model(batch['input_ids'].to(CONFIG['device']), batch['attention_mask'].to(CONFIG['device']))
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                targets.extend(batch['labels'].cpu().numpy())
                if curr_w is None: curr_w = w[0].detach().cpu().numpy().tolist()

        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='weighted')
        avg_loss = total_loss / len(train_loader)
        scheduler.step(acc)

        # History Update
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(acc)
        history['val_f1'].append(f1)
        history['weight_evolution'].append(curr_w)

        # Early Stopping
        if acc > best_val_acc:
            best_val_acc = acc
            best_weights = curr_w
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE: break
    
    return best_weights, best_val_acc, history

# ==============================================================================
# 6. MAIN (AGORA COM LOOP DE MODELOS E SEEDS)
# ==============================================================================
if __name__ == "__main__":
    
    # Loop Principal: Itera sobre cada modelo da lista
    for model_name in MODELS_TO_TEST:
        print(f"\n\n{'█'*60}")
        print(f"🤖 INICIANDO TESTES PARA O MODELO: {model_name}")
        print(f"{'█'*60}")
        
        # 1. Atualiza Configuração Global
        CONFIG['model_name'] = model_name
        
        # 2. Cria pasta específica para os resultados deste modelo
        safe_name = model_name.replace('/', '_')
        MODEL_SAVE_DIR = os.path.join(BASE_PATH, f'results_cap3_{safe_name}')
        if not os.path.exists(MODEL_SAVE_DIR): os.makedirs(MODEL_SAVE_DIR)
        
        # Dicionários para guardar as Médias Finais deste modelo
        final_averaged_weights = {} 
        final_accuracies_text = {}
        all_histories_dump = {}

        # Loop Secundário: Itera sobre as Tasks
        for task in CONFIG['tasks']:
            print(f"\n{'-'*40}")
            print(f"Task: {task} | Modelo: {model_name}")
            print(f"{'-'*40}")

            task_weights_runs = []
            task_acc_runs = []
            
            # Loop Terciário: Seeds (Robustez Estatística)
            for i, seed in enumerate(SEEDS):
                print(f"   >>> Seed {seed} ({i+1}/{len(SEEDS)})")
                
                # Configura seed antes de tudo
                set_seed(seed)
                
                # Treina
                weights, acc, history = train_task_engine(task)
                
                if weights is not None:
                    task_weights_runs.append(weights)
                    task_acc_runs.append(acc)
                    # Salva histórico com chave única: Task_Seed
                    all_histories_dump[f"{task}_seed_{seed}"] = history
                else:
                    print(f"   ❌ Falha na seed {seed}")

            # --- Processamento Pós-Task (Média das 5 seeds) ---
            if len(task_weights_runs) > 0:
                weights_np = np.array(task_weights_runs)
                acc_np = np.array(task_acc_runs)
                
                # Estatísticas
                mean_weights = np.mean(weights_np, axis=0)
                std_weights = np.std(weights_np, axis=0)
                mean_acc = np.mean(acc_np)
                std_acc = np.std(acc_np)
                
                print(f"\n   📊 Resultado {task}: {mean_acc:.4f} ± {std_acc:.4f}")
                
                # Guarda Média para o JSON Final
                final_averaged_weights[task] = mean_weights
                final_accuracies_text[task] = f"{mean_acc:.4f} ± {std_acc:.4f}"
                
                # Salva CSV bruto com todas as seeds para essa task
                df_raw = pd.DataFrame(weights_np, columns=[f"L{k+1}" for k in range(len(mean_weights))])
                df_raw['Seed'] = SEEDS[:len(task_weights_runs)]
                df_raw['Acc'] = acc_np
                df_raw.to_csv(os.path.join(MODEL_SAVE_DIR, f"raw_data_{task}.csv"), index=False)

        # ======================================================================
        # GERAÇÃO DE RESULTADOS POR MODELO
        # ======================================================================
        if len(final_averaged_weights) > 0:
            print(f"\n=== Gerando Relatórios para {model_name} ===")
            
            # 1. Tabela de Acurácias
            df_acc = pd.DataFrame(list(final_accuracies_text.items()), columns=['Task', 'Acc (Mean ± Std)'])
            df_acc.to_csv(os.path.join(MODEL_SAVE_DIR, 'final_accuracies.csv'), index=False)
            
            # 2. JSON de Pesos Médios e Heatmap
            df_weights = pd.DataFrame(final_averaged_weights).T
            # Ajusta nomes das colunas (L1, L2...)
            df_weights.columns = [f"L{i+1}" for i in range(df_weights.shape[1])]
            
            # Médias Universais
            df_weights.loc['AVG_ALL'] = df_weights.mean(axis=0)
            if 'NLI' in df_weights.index:
                df_weights.loc['AVG_SENTEVAL'] = df_weights.drop(['NLI', 'AVG_ALL'], errors='ignore').mean(axis=0)
            
            # Salva JSON
            json_ready = df_weights.T.to_dict(orient='list')
            with open(os.path.join(MODEL_SAVE_DIR, 'dynamic_weights_mean.json'), 'w') as f:
                json.dump(json_ready, f, indent=4)
            
            # Salva Históricos Completos
            hist_cleaned = {k: {m: clean_for_json(v) for m, v in h.items()} for k, h in all_histories_dump.items()}
            with open(os.path.join(MODEL_SAVE_DIR, 'all_histories.json'), 'w') as f:
                json.dump(hist_cleaned, f)

            # 3. Plot Heatmap (Baseado na Média das Seeds)
            plt.figure(figsize=(12, 10))
            sns.heatmap(df_weights, annot=True, fmt=".3f", cmap="YlGnBu")
            plt.title(f"Pesos Médios das Camadas - {model_name} (5 Seeds)")
            plt.tight_layout()
            plt.savefig(os.path.join(MODEL_SAVE_DIR, "heatmap_mean_weights.png"), dpi=300)
            plt.close()
            
            print(f"✅ Resultados salvos em: {MODEL_SAVE_DIR}")
        else:
            print(f"❌ Nenhuma tarefa completada para {model_name}")

    print("\n🏁 EXECUÇÃO COMPLETA DE TODOS OS MODELOS.")