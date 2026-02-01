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
import json
import warnings
from tqdm.auto import tqdm
import argparse

# Ignora avisos
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append('/home/yansoaresdell/research_embeddings_fine_tuning')

NLI_CSV_PATH = '/home/yansoaresdell/research_embeddings_fine_tuning/data/nli_optimized_for_layer_search_50k.csv'
SENTEVAL_DATA_PATH = '/home/yansoaresdell/research_embeddings_fine_tuning/data'

from senteval.binary import CREval, MREval, MPQAEval, SUBJEval
from senteval.trec import TRECEval
from senteval.sst import SSTEval
from senteval.mrpc import MRPCEval

SENTEVAL_AVAILABLE = True

#MODELS_TO_TEST = [
#    'microsoft/deberta-v3-base',
#    #'sentence-transformers/all-mpnet-base-v2',
#    #'answerdotai/ModernBERT-base',
#    #'microsoft/deberta-v3-large',
#    #'google-bert/bert-base-uncased',
#    #'FacebookAI/roberta-base',
#]

SEEDS = [42, 0, 1234, 2025, 999]

# ==============================================================================
# 1. HIPERPARÂMETROS GERAIS
# ==============================================================================
MAX_EPOCHS = 50
PATIENCE = 4

CONFIG = {
    'epochs': MAX_EPOCHS,
    'lr': 1e-3,
    'max_len': 128,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # Apenas tasks de classificação
    'tasks': ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC', 'NLI'],
    #'tasks': ['CR'],
    'seed_size': len(SEEDS)
}

if CONFIG['device'] == 'cpu':
    print("⚠️⚠️⚠️ ATENÇÃO: VOCÊ ESTÁ RODANDO EM CPU! VAI DEMORAR MUITO! ⚠️⚠️⚠️")
else:
    print(f"🚀 Rodando no dispositivo: {CONFIG['device']}")
    # Print extra para confirmar que viu a RTX 8000
    print(f"   Placa detectada: {torch.cuda.get_device_name(0)}")

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
    def __init__(self, model_name, num_classes, pooling_type): # Novo parâmetro
        super().__init__()
        self.pooling_type = pooling_type
        self.backbone = AutoModel.from_pretrained(model_name, output_hidden_states=True, return_dict=True)
        self.config = AutoConfig.from_pretrained(model_name)

        # Lógica de detecção de camadas (mantida)
        if hasattr(self.config, "num_hidden_layers"):
            self.n_layers = self.config.num_hidden_layers
        elif hasattr(self.config, "n_layers"):
            self.n_layers = self.config.n_layers
        else:
            self.n_layers = 22 
            
        print(f"Modelo {model_name} ({pooling_type}) inicializado com {self.n_layers} camadas.")
        
        for param in self.backbone.parameters(): 
            param.requires_grad = False
        self.backbone.eval()    

        self.fusion = DynamicFusionLayer(num_layers=self.n_layers)
        
        # Ajuste do tamanho da entrada do classificador
        classifier_input_size = self.config.hidden_size
        if pooling_type == 'cls_avg':
            classifier_input_size = self.config.hidden_size * 2
            
        self.classifier = nn.Linear(classifier_input_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        with torch.inference_mode():
            outputs = self.backbone(input_ids, attention_mask=attention_mask)
        
        # 1. Fusão das camadas (all_hidden_states)
        fused_sequence, weights = self.fusion(outputs.hidden_states)

        # 2. Estratégias de Pooling
        if self.pooling_type == 'cls':
            # Pega apenas o vetor do token [CLS] (índice 0)
            pooled_output = fused_sequence[:, 0, :]

        elif self.pooling_type == 'avg':
            # Média ponderada (sua implementação original)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(fused_sequence.size()).float()
            sum_embeddings = torch.sum(fused_sequence * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
            pooled_output = sum_embeddings / sum_mask

        elif self.pooling_type == 'cls_avg':
            # CLS
            cls_token = fused_sequence[:, 0, :]
            # AVG
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(fused_sequence.size()).float()
            sum_embeddings = torch.sum(fused_sequence * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
            avg_pooling = sum_embeddings / sum_mask
            # Concatena ambos: dimensão final será hidden_size * 2
            pooled_output = torch.cat((cls_token, avg_pooling), dim=-1)

        # 3. Classificação
        logits = self.classifier(self.dropout(pooled_output))
        return logits, weights

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()

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
        # 1. Tokenização
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

        # 2. Labels (Sempre Long para Classificação)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ==============================================================================
# 4. LOADERS & UTILS
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

def load_data_nli(csv_path, seed=42):
    if not os.path.exists(csv_path): return None, None, None, None
    df = pd.read_csv(csv_path)
    X = list(zip(df['premise'].astype(str), df['hypothesis'].astype(str)))
    y = df['label'].values
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

def load_data_senteval(task_name, seed=42):
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
def train_task_engine(task_name, seed_idx=0, total_seeds=5): # Adicionei argumentos para o log ficar bonito
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

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], num_workers=4, pin_memory=True, persistent_workers=True)

    model = TaskSpecificModel(CONFIG['model_name'], num_classes=num_classes, pooling_type=CONFIG['pooling_type']).to(CONFIG['device'])

    # 1. Separa os parâmetros
    fusion_params = list(model.fusion.parameters())
    classifier_params = list(model.classifier.parameters())

    # 2. Define o otimizador com dicionários de parâmetros (Parameter Groups)
    optimizer = optim.AdamW([
    {
        'params': fusion_params, 
        'lr': CONFIG['lr'],
        'weight_decay': 0.0
    },
    {
        'params': classifier_params, 
        'lr': CONFIG['lr'],
        'weight_decay': 0.01
    }
    ])

    #optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr'], weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_acc': [], 'val_f1': [], 'weight_evolution': []}
    best_val_metric = -1.0; best_weights = None; patience_counter = 0

    # --- TQDM CONFIG ---
    # Cria uma barra de progresso para as ÉPOCAS
    epochs_bar = tqdm(range(CONFIG['epochs']), desc=f"Seed {seed_idx+1}/{total_seeds} | {task_name}", leave=False)

    scaler = torch.cuda.amp.GradScaler(enabled=(CONFIG['device'] == 'cuda'))

    for epoch in epochs_bar:
        # Train
        model.train()
        total_loss = 0
       
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(CONFIG['device'] == 'cuda')):
                logits, _ = model(batch['input_ids'].to(CONFIG['device'], non_blocking=True),
                                batch['attention_mask'].to(CONFIG['device'], non_blocking=True))
                loss = criterion(logits, batch['labels'].to(CONFIG['device'], non_blocking=True))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(model.fusion.parameters()) + list(model.classifier.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
        
        # Val
        model.eval()
        preds, targets = [], []
        curr_w = None
        with torch.no_grad():
            for batch in val_loader:
                logits, w = model(batch['input_ids'].to(CONFIG['device'], non_blocking=True), batch['attention_mask'].to(CONFIG['device'], non_blocking=True))
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                targets.extend(batch['labels'].cpu().numpy())
                if curr_w is None: curr_w = w.detach().cpu().numpy().tolist()

        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='weighted')
        avg_loss = total_loss / len(train_loader)
        scheduler.step(acc)

        history['train_loss'].append(avg_loss)
        history['val_acc'].append(acc)
        history['val_f1'].append(f1)
        history['weight_evolution'].append(curr_w)

        if curr_w is None:
            entropy = float("nan")
        else:
            w_entropy = np.array(curr_w, dtype=np.float64)
            entropy = -(w_entropy * np.log(w_entropy + 1e-12)).sum()

        epochs_bar.set_postfix({'Loss': f"{avg_loss:.4f}", 'Acc': f"{acc:.4f}", 'H': f"{entropy:.2f}"})

        metric = acc  # SentEval: selecione por Accuracy
        if metric > best_val_metric:
            best_val_metric = metric
            best_weights = curr_w
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                epochs_bar.write(f"   -> Early Stopping na época {epoch+1}") # Usa .write pra não quebrar a barra
                break
    
    return best_weights, best_val_metric, history

# ==============================================================================
# 6. MAIN (LOOP DE MODELOS E SEEDS)
# ==============================================================================
if __name__ == "__main__":   
   
    parser = argparse.ArgumentParser(description='Treinamento Task-Specific com Fusão de Camadas')
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-base', help='Nome do modelo no HuggingFace')
    parser.add_argument('--pooling_type', type=str, default='avg', choices=['avg', 'cls', 'cls_avg'], help='Estratégia de pooling')
    parser.add_argument('--base_path', type=str, default='cap3', help='Estratégia de pooling')
    #parser.add_argument('--batch_size', type=int, default=128, help='Tamanho do batch')
   
    args = parser.parse_args()

    # Atualiza o CONFIG global com os argumentos passados
    CONFIG['model_name'] = args.model_name
    CONFIG['pooling_type'] = args.pooling_type
    #CONFIG['batch_size'] = args.batch_size
    BASE_PATH = args.base_path

    # --- AQUI ESTÁ O USO DA SUA RTX 8000 ---
    if 'large' in CONFIG['model_name'].lower():
        CONFIG['batch_size'] = 64  # Large cabe tranquilo com 128
    else:
        CONFIG['batch_size'] = 128  # Base voa com 256
        
    print(f"\n{'█'*60}")
    print(f"🤖 MODELO: {CONFIG['model_name']}")
    print(f"🎯 POOLING: {CONFIG['pooling_type']}")
    print(f"📦 BATCH: {CONFIG['batch_size']}")
    print(f"{'█'*60}")
    
    
    # --- NOVO: PRINT DE VERIFICAÇÃO ---
    print(f"⏳ Verificando e carregando {CONFIG['model_name']} na memória...")
    try:
        AutoConfig.from_pretrained(CONFIG['model_name'])
        print(f"✅ Modelo verificado! Iniciando experimentos...")
    except Exception as e:
        print(f"❌ Erro ao carregar {CONFIG['model_name']}. Pulando...")
    
    safe_name = CONFIG['model_name'].replace('/', '_')
    MODEL_SAVE_DIR = os.path.join(BASE_PATH, f'results_{safe_name}', CONFIG['pooling_type'])
    if not os.path.exists(MODEL_SAVE_DIR): 
        os.makedirs(MODEL_SAVE_DIR)
    
    final_averaged_weights = {} 
    final_std_weights = {}
    final_accuracies_text = {}
    all_histories_dump = {}

    # --- BARRA GERAL DE TASKS ---
    task_bar = tqdm(CONFIG['tasks'], desc=f"Progresso Geral ({CONFIG['model_name']})")

    for task in task_bar:
        # Atualiza a descrição da barra geral
        task_bar.set_description(f"Rodando {task}...")

        task_weights_runs = []
        task_acc_runs = []
        
        for i, seed in enumerate(SEEDS):
            set_seed(seed)
            # Passamos o índice da seed para o tqdm interno ficar bonito
            weights, acc, history = train_task_engine(task, seed_idx=i, total_seeds=len(SEEDS))
            
            if weights is not None:
                task_weights_runs.append(weights)
                task_acc_runs.append(acc)
                all_histories_dump[f"{task}_seed_{seed}"] = history
            else:
                print(f"   ❌ Falha na seed {seed}")

        # --- Pós-processamento (Média das seeds) ---
        if len(task_weights_runs) > 0:
            weights_np = np.array(task_weights_runs)
            acc_np = np.array(task_acc_runs)
            
            mean_weights = np.mean(weights_np, axis=0)
            std_weights = np.std(weights_np, axis=0)
            mean_acc = np.mean(acc_np)
            std_acc = np.std(acc_np)
            
            # Usa task_bar.write para imprimir sem quebrar a barra de progresso
            task_bar.write(f"   📊 Resultado {task}: {mean_acc:.4f} ± {std_acc:.4f}")
            
            final_averaged_weights[task] = mean_weights
            final_std_weights[task] = std_weights
            final_accuracies_text[task] = f"{mean_acc:.4f} ± {std_acc:.4f}"
            
            # --- CORREÇÃO DE DEBUG ---
            print(f"\n[DEBUG] Shape de weights_np: {weights_np.shape}")
            
            # Se weights_np for 1D (apenas [fold1, fold2...]) em vez de 2D ([fold1_layers, fold2_layers...])
            # Precisamos garantir que ele seja tratado como matriz
            if len(weights_np.shape) == 1:
                print("[AVISO] weights_np está 1D. O modelo pode estar com apenas 1 peso de camada ou houve achatamento.")
                # Força o reshape se necessário ou alerta
                if hasattr(mean_weights, 'shape') and mean_weights.shape == ():
                    # Se mean_weights virou um escalar (float64 sem dimensão)
                    mean_weights = np.array([mean_weights])
            
            # Garante que num_layers é um inteiro válido
            if isinstance(mean_weights, (float, np.floating)):
                num_layers_detected = 1
                mean_weights = [mean_weights] # Converte para lista
            else:
                num_layers_detected = len(mean_weights)

            print(f"[DEBUG] Camadas detectadas para DataFrame: {num_layers_detected}")

            # Criação do DataFrame Blindada
            df_raw = pd.DataFrame(weights_np, columns=[f"L{k+1}" for k in range(num_layers_detected)])
            
            # Para o df_mean, precisamos garantir que seja uma linha (DataFrame requer 2D ou index)
            df_mean = pd.DataFrame([mean_weights], columns=[f"L{k+1}" for k in range(num_layers_detected)])
            # -------------------------

    # ======================================================================
    # GERAÇÃO DE RESULTADOS POR MODELO (Sem alterações aqui)
    # ======================================================================
    if len(final_averaged_weights) > 0:
        print(f"\n=== Gerando Relatórios para {CONFIG['model_name']} ===")
        
        df_acc = pd.DataFrame(list(final_accuracies_text.items()), columns=['Task', 'Acc (Mean ± Std)'])
        df_acc.to_csv(os.path.join(MODEL_SAVE_DIR, 'final_accuracies.csv'), index=False)
        
        df_weights = pd.DataFrame(final_averaged_weights).T
        df_weights.columns = [f"L{i+1}" for i in range(df_weights.shape[1])]
        
        df_weights.loc['AVG_ALL'] = df_weights.mean(axis=0)
        if 'NLI' in df_weights.index:
            df_weights.loc['AVG_SENTEVAL'] = df_weights.drop(['NLI', 'AVG_ALL'], errors='ignore').mean(axis=0)
        
        json_ready = df_weights.T.to_dict(orient='list')
        json_ready = {k: [float(x) for x in v] for k, v in json_ready.items()}
        with open(os.path.join(MODEL_SAVE_DIR, 'dynamic_weights_mean.json'), 'w') as f:
            json.dump(json_ready, f, indent=4)
        
        hist_cleaned = {k: {m: clean_for_json(v) for m, v in h.items()} for k, h in all_histories_dump.items()}
        with open(os.path.join(MODEL_SAVE_DIR, 'all_histories.json'), 'w') as f:
            json.dump(hist_cleaned, f)

        plt.figure(figsize=(18, 12))
        sns.heatmap(df_weights, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title(f"Pesos Médios das Camadas - {CONFIG['model_name']} - {CONFIG['seed_size']}")
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_SAVE_DIR, "heatmap_mean_weights.png"), dpi=300)
        plt.close()

        # ==========================================================
        # 3. [NOVO] SALVANDO DESVIO PADRÃO DOS PESOS
        # ==========================================================
        df_std = pd.DataFrame(final_std_weights).T
        df_std.columns = [f"L{i+1}" for i in range(df_std.shape[1])]
        
        # Salva CSV/JSON do Desvio Padrão
        df_std.to_csv(os.path.join(MODEL_SAVE_DIR, 'dynamic_weights_std.csv'))
        
        # [OPCIONAL] Heatmap do Desvio Padrão (Isso é muito útil!)
        # Mostra onde o modelo está "indeciso" entre as seeds
        plt.figure(figsize=(18, 12))
        sns.heatmap(df_std, annot=True, fmt=".3f", cmap="Reds", annot_kws={"size": 10}) # Usei vermelho pra indicar "instabilidade"
        plt.title(f"DESVIO PADRÃO dos Pesos (Instabilidade) - {CONFIG['model_name']}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_SAVE_DIR, "heatmap_std_weights.png"), dpi=300)
        plt.close()
        
        print(f"✅ Resultados salvos em: {MODEL_SAVE_DIR}")

    print("\n🏁 EXECUÇÃO COMPLETA DE TODOS OS MODELOS.")