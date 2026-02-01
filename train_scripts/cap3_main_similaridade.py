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
from scipy.stats import pearsonr

# Ignora avisos
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append('/home/yansoaresdell/research_embeddings_fine_tuning')

BASE_PATH = 'final_results_cap3_similaridade'
SENTEVAL_DATA_PATH = '/home/yansoaresdell/research_embeddings_fine_tuning/data'

MODELS_TO_TEST = [
    'sentence-transformers/all-mpnet-base-v2',
    'microsoft/deberta-v3-base',    
    #'answerdotai/ModernBERT-base',
    #'microsoft/deberta-v3-large',
    #'google-bert/bert-base-uncased',
    #'FacebookAI/roberta-base',
]

SEEDS = [42, 0, 1234, 2025, 999]

# ==============================================================================
# 1. HIPERPARÂMETROS GERAIS (espelhado)
# ==============================================================================
MAX_EPOCHS = 50
PATIENCE = 4

CONFIG = {
    'epochs': MAX_EPOCHS,
    'lr': 1e-3,
    'max_len': 128,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # Tasks de similaridade (mesmas do seu main.py)
    'tasks': ['SICKRelatedness', 'STSBenchmark']
}

if CONFIG['device'] == 'cpu':
    print("⚠️⚠️⚠️ ATENÇÃO: VOCÊ ESTÁ RODANDO EM CPU! VAI DEMORAR MUITO! ⚠️⚠️⚠️")
else:
    print(f"🚀 Rodando no dispositivo: {CONFIG['device']}")
    print(f"   Placa detectada: {torch.cuda.get_device_name(0)}")

# ==============================================================================
# 2. MODELO (FUSÃO + POOLING AVG, espelhado)
# ==============================================================================
class DynamicFusionLayer(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, all_hidden_states):
        stacked_layers = torch.stack(all_hidden_states[1:], dim=1)  # [B, L, T, H]
        norm_weights = F.softmax(self.layer_weights, dim=0)          # [L]
        weights_reshaped = norm_weights.view(1, -1, 1, 1)            # [1, L, 1, 1]
        fused_embedding = (stacked_layers * weights_reshaped).sum(dim=1)  # [B, T, H]
        return fused_embedding, norm_weights


class STSTaskModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, output_hidden_states=True, return_dict=True)
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
        self.backbone.eval()

        self.fusion = DynamicFusionLayer(num_layers=self.n_layers)

    def forward(self, input_ids, attention_mask):
        with torch.inference_mode():
            outputs = self.backbone(input_ids, attention_mask=attention_mask)

        fused_sequence, weights = self.fusion(outputs.hidden_states)

        mask = attention_mask.unsqueeze(-1).expand(fused_sequence.size()).float()
        pooled = torch.sum(fused_sequence * mask, 1) / torch.clamp(mask.sum(1), min=1e-4)

        return pooled, weights

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()



# ==============================================================================
# 3. DATASET
# ==============================================================================
class STSPairDataset(Dataset):
    def __init__(self, s1, s2, scores, tokenizer, max_len):
        self.s1 = s1
        self.s2 = s2
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.s1)

    def __getitem__(self, idx):
        a = self.tokenizer(
            str(self.s1[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        b = self.tokenizer(
            str(self.s2[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'a_ids': a['input_ids'].flatten(),
            'a_mask': a['attention_mask'].flatten(),
            'b_ids': b['input_ids'].flatten(),
            'b_mask': b['attention_mask'].flatten(),
            'label': torch.tensor(float(self.scores[idx]), dtype=torch.float32)
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

def pearson_loss(pred, target, eps=1e-8):
    pred = pred - pred.mean()
    target = target - target.mean()

    num = torch.sum(pred * target)
    den = torch.sqrt(torch.sum(pred ** 2) * torch.sum(target ** 2)) + eps
    return 1.0 - num / den

# ==============================================================================
# 5. LOADERS STS (compatível com arquivos SentEval)
# ==============================================================================
def load_sick_relatedness():
    base = os.path.join(SENTEVAL_DATA_PATH, "downstream", "SICK")

    train_path = os.path.join(base, "SICK_train.txt")
    dev_path   = os.path.join(base, "SICK_trial.txt")

    if not (os.path.exists(train_path) and os.path.exists(dev_path)):
        print(f"[ERRO] Arquivos do SICK não encontrados em {base}")
        return None, None

    def _read_sick(fpath):
        rows = []
        with open(fpath, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            for line in f:
                cols = line.strip().split("\t")
                if len(cols) < len(header):
                    continue

                row = dict(zip(header, cols))

                # nomes padrão do SICK
                s1 = row.get("sentence_A")
                s2 = row.get("sentence_B")
                score = row.get("relatedness_score")

                if s1 is None or s2 is None or score is None:
                    continue

                try:
                    score = float(score)
                except:
                    continue

                rows.append((s1, s2, score))

        df = pd.DataFrame(rows, columns=["s1", "s2", "score"])

        # normaliza 1..5 -> 0..1
        if df["score"].max() > 1.0:
            df["score"] = (df["score"] - 1.0) / 4.0

        return df

    train_df = _read_sick(train_path)
    dev_df   = _read_sick(dev_path)

    if len(train_df) == 0 or len(dev_df) == 0:
        print("[ERRO] SICK train/dev vazios")
        return None, None

    return train_df, dev_df

def load_stsbenchmark():
    base = os.path.join(SENTEVAL_DATA_PATH, "downstream", "stsbenchmark")

    def _read(path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 7:
                    continue
                score = float(cols[4])
                s1 = cols[5]
                s2 = cols[6]
                rows.append((s1, s2, score))

        df = pd.DataFrame(rows, columns=["s1", "s2", "score"])
        df["score"] = df["score"] / 5.0
        return df

    train_df = _read(os.path.join(base, "sts-train.csv"))
    dev_df   = _read(os.path.join(base, "sts-dev.csv"))

    return train_df, dev_df

def load_sts_task(task_name):
    if task_name == "STSBenchmark":
        return load_stsbenchmark()

    if task_name == "SICKRelatedness":
        return load_sick_relatedness()

    return None, None


# ==============================================================================
# 6. ENGINE DE TREINO (igual ao de classificação: AMP, scheduler, early stop)
# ==============================================================================
def train_sts_engine(task_name, seed_idx=0, total_seeds=5):
    train_df, dev_df = load_sts_task(task_name)
    if train_df is None:
        print(f"❌ Dados não encontrados para {task_name}. Pulando.")
        return None, -1.0, None

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

    train_ds = STSPairDataset(train_df['s1'].values, train_df['s2'].values, train_df['score'].values, tokenizer, CONFIG['max_len'])
    dev_ds   = STSPairDataset(dev_df['s1'].values, dev_df['s2'].values, dev_df['score'].values, tokenizer, CONFIG['max_len'])

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    dev_loader = DataLoader(dev_ds, batch_size=CONFIG['batch_size'],
                            num_workers=4, pin_memory=True, persistent_workers=True)

    model = STSTaskModel(CONFIG['model_name']).to(CONFIG['device'])

    # ✅ ALINHADO AO CLASSIFICADOR: WD=0 na fusão
    fusion_params = list(model.fusion.parameters())
    optimizer = optim.AdamW([
        {'params': fusion_params, 'lr': CONFIG['lr'], 'weight_decay': 0.0},
    ])

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion_mse = nn.MSELoss()

    history = {'train_loss': [], 'dev_pearson': [], 'weight_evolution': [], 'entropy': []}
    best_dev = -1.0
    best_weights = None
    patience_counter = 0

    epochs_bar = tqdm(range(CONFIG['epochs']), desc=f"Seed {seed_idx+1}/{total_seeds} | {task_name}", leave=False)
    scaler = torch.cuda.amp.GradScaler(enabled=(CONFIG['device'] == 'cuda'))

    for epoch in epochs_bar:
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            a_ids = batch['a_ids'].to(CONFIG['device'], non_blocking=True)
            a_msk = batch['a_mask'].to(CONFIG['device'], non_blocking=True)
            b_ids = batch['b_ids'].to(CONFIG['device'], non_blocking=True)
            b_msk = batch['b_mask'].to(CONFIG['device'], non_blocking=True)
            y     = batch['label'].to(CONFIG['device'], non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(CONFIG['device'] == 'cuda')):
                va, _ = model(a_ids, a_msk)
                vb, _ = model(b_ids, b_msk)

                pred = F.cosine_similarity(va, vb).clamp(-1.0, 1.0)
                pred01 = (pred + 1.0) / 2.0

                mse = criterion_mse(pred01, y)
                loss = 0.5 * pearson_loss(pred01, y) + 0.5 * mse

            scaler.scale(loss).backward()

            # ✅ ALINHADO: clip (com unscale)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(fusion_params, 1.0)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        # DEV
        model.eval()
        all_preds, all_labels = [], []
        curr_w = None

        with torch.no_grad():
            for batch in dev_loader:
                a_ids = batch['a_ids'].to(CONFIG['device'], non_blocking=True)
                a_msk = batch['a_mask'].to(CONFIG['device'], non_blocking=True)
                b_ids = batch['b_ids'].to(CONFIG['device'], non_blocking=True)
                b_msk = batch['b_mask'].to(CONFIG['device'], non_blocking=True)

                va, w = model(a_ids, a_msk)
                vb, _ = model(b_ids, b_msk)

                pred = F.cosine_similarity(va, vb).clamp(-1.0, 1.0)
                pred01 = (pred + 1.0) / 2.0

                all_preds.extend(pred01.detach().cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())

                if curr_w is None:
                    curr_w = w.detach().cpu().numpy().tolist()

        dev_pearson = pearsonr(all_preds, all_labels)[0]
        avg_loss = total_loss / max(1, len(train_loader))

        scheduler.step(dev_pearson)

        history['train_loss'].append(avg_loss)
        history['dev_pearson'].append(dev_pearson)
        history['weight_evolution'].append(curr_w)

        # ✅ Log de entropia (igual ao classificador)
        if curr_w is None:
            entropy = float("nan")
        else:
            w_arr = np.array(curr_w, dtype=np.float64)
            entropy = -(w_arr * np.log(w_arr + 1e-12)).sum()
        history['entropy'].append(entropy)

        epochs_bar.set_postfix({'Loss': f"{avg_loss:.4f}", 'Pearson': f"{dev_pearson:.4f}", 'H': f"{entropy:.2f}"})

        if dev_pearson > best_dev:
            best_dev = dev_pearson
            best_weights = curr_w
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                epochs_bar.write(f"   -> Early Stopping na época {epoch+1}")
                break

    return best_weights, best_dev, history

# ==============================================================================
# 7. MAIN (loop modelos + tasks + seeds, igual ao de classificação)
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
            print(f"✅ Modelo verificado! Iniciando experimentos...")
        except Exception:
            print(f"❌ Erro ao carregar {model_name}. Pulando...")
            continue

        safe_name = model_name.replace('/', '_')
        MODEL_SAVE_DIR = os.path.join(BASE_PATH, f'results_cap3_{safe_name}')
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

        final_averaged_weights = {}
        final_std_weights = {}
        final_scores_text = {}
        all_histories_dump = {}

        task_bar = tqdm(CONFIG['tasks'], desc=f"Progresso Geral ({model_name})")

        for task in task_bar:
            task_bar.set_description(f"Rodando {task}...")

            task_weights_runs = []
            task_score_runs = []

            for i, seed in enumerate(SEEDS):
                set_seed(seed)
                weights, score, history = train_sts_engine(task, seed_idx=i, total_seeds=len(SEEDS))

                if weights is not None:
                    task_weights_runs.append(weights)
                    task_score_runs.append(score)
                    all_histories_dump[f"{task}_seed_{seed}"] = history
                else:
                    print(f"   ❌ Falha na seed {seed}")

            if len(task_weights_runs) > 0:
                weights_np = np.array(task_weights_runs)
                score_np = np.array(task_score_runs)

                mean_weights = np.mean(weights_np, axis=0)
                std_weights = np.std(weights_np, axis=0)
                mean_score = np.mean(score_np)
                std_score = np.std(score_np)

                task_bar.write(f"   📊 Resultado {task}: {mean_score:.4f} ± {std_score:.4f}")

                final_averaged_weights[task] = mean_weights
                final_std_weights[task] = std_weights
                final_scores_text[task] = f"{mean_score:.4f} ± {std_score:.4f}"

        # ======================================================================
        # SALVAMENTO (mesmo padrão do classificador)
        # ======================================================================
        if len(final_averaged_weights) > 0:
            print(f"\n=== Gerando Relatórios para {model_name} ===")

            df_score = pd.DataFrame(list(final_scores_text.items()), columns=['Task', 'Pearson (Mean ± Std)'])
            df_score.to_csv(os.path.join(MODEL_SAVE_DIR, 'final_pearsons.csv'), index=False)

            df_weights = pd.DataFrame(final_averaged_weights).T
            df_weights.columns = [f"L{i+1}" for i in range(df_weights.shape[1])]

            # também cria médias como no classificador
            df_weights.loc['AVG_ALL'] = df_weights.mean(axis=0)

            json_ready = df_weights.T.to_dict(orient='list')
            json_ready = {k: [float(x) for x in v] for k, v in json_ready.items()}

            with open(os.path.join(MODEL_SAVE_DIR, 'dynamic_weights_mean.json'), 'w') as f:
                json.dump(json_ready, f, indent=4)

            hist_cleaned = {k: {m: clean_for_json(v) for m, v in h.items()} for k, h in all_histories_dump.items()}
            with open(os.path.join(MODEL_SAVE_DIR, 'all_histories.json'), 'w') as f:
                json.dump(hist_cleaned, f)

            plt.figure(figsize=(18, 12))
            sns.heatmap(df_weights, annot=True, fmt=".3f", cmap="YlGnBu")
            plt.title(f"Pesos Médios das Camadas (STS) - {model_name} (Seeds)")
            plt.tight_layout()
            plt.savefig(os.path.join(MODEL_SAVE_DIR, "heatmap_mean_weights.png"), dpi=300)
            plt.close()

            df_std = pd.DataFrame(final_std_weights).T
            df_std.columns = [f"L{i+1}" for i in range(df_std.shape[1])]
            df_std.to_csv(os.path.join(MODEL_SAVE_DIR, 'dynamic_weights_std.csv'))

            plt.figure(figsize=(18, 12))
            sns.heatmap(df_std, annot=False, cmap="Reds")
            plt.title(f"DESVIO PADRÃO dos Pesos (STS) - {model_name}", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(MODEL_SAVE_DIR, "heatmap_std_weights.png"), dpi=300)
            plt.close()

            print(f"✅ Resultados salvos em: {MODEL_SAVE_DIR}")

    print("\n🏁 EXECUÇÃO COMPLETA DE TODOS OS MODELOS (STS).")
