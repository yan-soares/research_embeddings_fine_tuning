import senteval
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging
import os
import json
import time
import warnings
from sklearn.exceptions import ConvergenceWarning
from nltk.corpus import stopwords
import numpy as np

import functions_code

warnings.filterwarnings("ignore", category=ConvergenceWarning)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# ==============================================================================
# CLASSES
# ==============================================================================
class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention_projection = nn.Linear(in_dim, in_dim)
        self.context_vector = nn.Linear(in_dim, 1, bias=False)

    def forward(self, hidden_states, mask):
        # 1. Calcula energia (W * h)
        weights = torch.tanh(self.attention_projection(hidden_states))
        # 2. Compara com contexto (u * energy)
        weights = self.context_vector(weights).squeeze(-1)
        # 3. Mascara padding (-inf)
        weights = weights.masked_fill(mask == 0, -1e4)
        # 4. Softmax (Probabilidades)
        weights = F.softmax(weights, dim=1)
        # 5. Soma Ponderada
        weights_expanded = weights.unsqueeze(-1)
        weighted_embeddings = torch.sum(hidden_states * weights_expanded, dim=1)

        return weighted_embeddings

class SentenceEncoder:
    def __init__(self, model_name, device, dynamic_weights_path, attention_weights_path=None):
        self.device = device
        self.size_embedding = None
        self.pooling_strategy = None
        self.print_best_layers = None
        
        self.stopwords_set_ids = None
        self.cls_token_id = None
        self.sep_token_id = None

        self.general_embeddings = {}
        self.list_poolings = None
        self.list_layers = None
        self.actual_layer = None
        self.current_task_name = None

        # --- Inicializa o modelo escolhido ---
        self.name_model, self.qtd_layers, self.tokenizer, self.model = functions_code.load_model(model_name, device)
        self._prepare_special_token_ids()

        # --- FUSÃO (CAP 3) ---
        self.dynamic_weights = {}
        self.dynamic_weights_path = dynamic_weights_path
        if self.dynamic_weights_path and os.path.exists(self.dynamic_weights_path):
            print(f"Carregando Pesos de Fusão de: {self.dynamic_weights_path}")
            with open(self.dynamic_weights_path, 'r') as f:
                self.dynamic_weights = json.load(f)
        else:
            if self.dynamic_weights_path:
                print(f"AVISO: Arquivo de fusão {self.dynamic_weights_path} não encontrado.")

        # --- ATENÇÃO (CAP 4) ---
        self.att_pool = AttentionPooling(self.model.config.hidden_size).to(self.device)
        self.all_attention_weights = {}
        if attention_weights_path and os.path.exists(attention_weights_path):
            print(f"Carregando Pesos de Atenção (Cap 4) de: {attention_weights_path}")
            self.all_attention_weights = torch.load(attention_weights_path, map_location=self.device)
        else:
            if attention_weights_path:
                print(f"AVISO: Arquivo de atenção {attention_weights_path} não encontrado.")
       
    def _prepare_special_token_ids(self):
        stopwords_list = stopwords.words('english')
        stopword_ids = self.tokenizer.convert_tokens_to_ids(stopwords_list)
        stopword_ids_filtered = [id for id in stopword_ids if id != self.tokenizer.unk_token_id]
        self.stopwords_set_ids = torch.tensor(stopword_ids_filtered, device=self.device)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

    def _encode(self, sentences, current_task, batch_size=16384): 
        self.current_task_name = current_task
        tokens = self.tokenizer(sentences, padding="longest", truncation=True, return_tensors="pt", max_length=self.model.config.max_position_embeddings)
        tokens = {key: val.to(self.device, non_blocking=True) for key, val in tokens.items()}
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch_tokens = {key: val[i:i+batch_size] for key, val in tokens.items()}
            with torch.inference_mode(), torch.amp.autocast('cuda'):
                output = self.model(**batch_tokens)
                embeddings = self._apply_pooling(output, batch_tokens['attention_mask'], batch_tokens['input_ids'])
            all_embeddings.append(embeddings)

        self.size_embedding = all_embeddings[0].shape 
        final_embeddings = torch.cat(all_embeddings, dim=0).to('cpu').numpy()
        del batch_tokens, output
        torch.cuda.empty_cache() 
        return final_embeddings

    def _create_combined_mask(self, input_ids, attention_mask, exclude_stopwords=False, exclude_cls_sep=False):
        combined_mask = attention_mask.clone()
        if exclude_stopwords:
            stopword_mask = torch.isin(input_ids, self.stopwords_set_ids, invert=True)
            combined_mask = combined_mask * stopword_mask
        if exclude_cls_sep:
            special_tokens_mask = (input_ids != self.cls_token_id) & (input_ids != self.sep_token_id)
            combined_mask = combined_mask * special_tokens_mask
        return combined_mask
    
    def _mean_pooling_exclude_cls_sep_and_stopwords(self, output, attention_mask, input_ids):
        mask = self._create_combined_mask(input_ids, attention_mask, exclude_stopwords=True, exclude_cls_sep=True)
        expanded_mask = mask.unsqueeze(-1)
        masked_embeddings = output * expanded_mask
        sum_embeddings = masked_embeddings.sum(dim=1)
        valid_token_counts = expanded_mask.sum(dim=1)
        return sum_embeddings / valid_token_counts.clamp(min=1e-9)

    def _sum_pooling_exclude_cls_sep_and_stopwords(self, output, attention_mask, input_ids):
        mask = self._create_combined_mask(input_ids, attention_mask, exclude_stopwords=True, exclude_cls_sep=True)
        expanded_mask = mask.unsqueeze(-1)
        return (output * expanded_mask).sum(dim=1)
    
    def _max_pooling_exclude_cls_sep_and_stopwords(self, output, attention_mask, input_ids):
        mask = self._create_combined_mask(input_ids, attention_mask, exclude_stopwords=True, exclude_cls_sep=True)
        expanded_mask = mask.unsqueeze(-1)
        masked_embeddings = output.masked_fill(expanded_mask == 0, -1e9)
        return masked_embeddings.max(dim=1)[0]
    
    def _simple_pooling(self, hidden_state, attention_mask, name_pooling, input_ids):
        match name_pooling:
            case "CLS":
                return hidden_state[:, 0, :]
            case "AVG":
                return ((hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
            case "SUM":
                return (hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1)
            case "MAX":
                return torch.max(hidden_state.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9), dim=1)[0]
            case "AVG-NS":
                return self._mean_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)
            case "SUM-NS":
                return self._sum_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)
            case "MAX-NS":
                return self._max_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)
            
            # --- OPÇÃO 1: ATTENTION-ENSEMBLE (A novidade) ---
            # Média de Especialistas: Tenta melhorar TREC/MRPC misturando conhecimentos
            case "ATTENTION-ENSEMBLE":
                source_weights = None
                if self.all_attention_weights:
                    # Define os especialistas
                    tasks_to_mix = ['SST2', 'TREC', 'MRPC']
                    
                    # Pega o que tem disponível no arquivo .pt
                    available_dicts = [self.all_attention_weights[t] for t in tasks_to_mix if t in self.all_attention_weights]
                    
                    if available_dicts:
                        # Clona o primeiro para começar a média
                        avg_weights = {k: v.clone() for k, v in available_dicts[0].items()}
                        
                        # Soma os outros
                        for i in range(1, len(available_dicts)):
                            for key in avg_weights:
                                avg_weights[key] += available_dicts[i][key]
                        
                        # Divide pelo total
                        for key in avg_weights:
                            avg_weights[key] = avg_weights[key] / len(available_dicts)
                            
                        source_weights = avg_weights
                
                # Aplica
                if source_weights is not None:
                    self.att_pool.load_state_dict(source_weights)
                return self.att_pool(hidden_state, attention_mask)

            # --- OPÇÃO 2: ATTENTION (O Padrão/Transfer) ---
            # Source Swapping: SST2 ensina todo mundo
            case "ATTENTION":
                source_weights = None
                if self.all_attention_weights:
                    # Lógica de Honestidade (SST2 <-> MR)
                    if self.current_task_name == 'SST2':
                        if 'MR' in self.all_attention_weights:
                            source_weights = self.all_attention_weights['MR']
                        else:
                            # Fallback se não tiver MR
                            av = [k for k in self.all_attention_weights.keys() if k != 'SST2']
                            if av: source_weights = self.all_attention_weights[av[0]]
                    else:
                        # Para todas as outras, usa SST2
                        if 'SST2' in self.all_attention_weights:
                            source_weights = self.all_attention_weights['SST2']
                        elif 'MR' in self.all_attention_weights:
                            source_weights = self.all_attention_weights['MR']

                if source_weights is not None:
                    self.att_pool.load_state_dict(source_weights)
                return self.att_pool(hidden_state, attention_mask)
                     
    def _get_pooling_result(self, hidden_state, attention_mask, name_pooling, name_agg, input_ids):
        self.print_best_layers =  "NORMAL"        
        name_pooling_split = name_pooling.split('+')
        # ... (Logica de concatenação mantida) ...
        match len(name_pooling_split):
            case 1:
                return self._simple_pooling(hidden_state, attention_mask, name_pooling_split[0], input_ids)
            case 2:
                return torch.cat((
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[0], input_ids),
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[1], input_ids)
                ), dim=1)
            case 3:
                return torch.cat((
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[0], input_ids),
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[1], input_ids),
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[2], input_ids)
                ), dim=1)
            case 4:
                return torch.cat((
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[0], input_ids),
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[1], input_ids),
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[2], input_ids),
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[3], input_ids)
                ), dim=1)

    # --- NOVO MÉTODO AUXILIAR (Adicione dentro da classe SentenceEncoder) ---
    def _compute_mean_weights(self, task_list):
        """Calcula a média dos pesos para uma lista de tarefas, ignorando as que não existem no JSON."""
        valid_vectors = []
        for task in task_list:
            # Garante que temos pesos para essa task
            if task in self.dynamic_weights:
                valid_vectors.append(np.array(self.dynamic_weights[task]))
        
        if not valid_vectors:
            return None
            
        # Calcula a média coluna por coluna (layer por layer)
        mean_vector = np.mean(valid_vectors, axis=0)
        return mean_vector.tolist()

    # --- MÉTODO _apply_pooling ATUALIZADO ---
    def _apply_pooling(self, output, attention_mask, input_ids):  
        hidden_states = output.hidden_states
        name_pooling = self.pooling_strategy.split("_")[0] 
        name_agg = self.pooling_strategy.split("_")[-1]    

        # Listas de definição para os cálculos de conjuntos
        senteval_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
        all_tasks_with_nli = senteval_tasks + ['NLI']

        # Lista das novas estratégias aceitas
        dynamic_strategies = [
            'IN-TASK', 'LEAVE-ONE-OUT-SENTEVAL', 'AVG-SENTEVAL', 
            'NLI', 'LEAVE-ONE-OUT-SENTEVAL-NLI', 'AVG-ALL'
        ]

        if name_agg in dynamic_strategies:
            weights = None
            
            # 1. IN-TASK (Usa os pesos da própria tarefa atual) [Igual ao antigo DYNAMIC]
            if name_agg == 'IN-TASK':
                if self.current_task_name in self.dynamic_weights:
                    weights = self.dynamic_weights[self.current_task_name]

            # 2. LEAVE-ONE-OUT-SENTEVAL (Média do SentEval - Task Atual)
            elif name_agg == 'LEAVE-ONE-OUT-SENTEVAL':
                target_tasks = [t for t in senteval_tasks if t != self.current_task_name]
                weights = self._compute_mean_weights(target_tasks)

            # 3. AVG-SENTEVAL (Média de todas do SentEval)
            elif name_agg == 'AVG-SENTEVAL':
                # Tenta pegar direto do JSON se existir, senão calcula
                if 'AVG_SENTEVAL' in self.dynamic_weights:
                    weights = self.dynamic_weights['AVG_SENTEVAL']
                else:
                    weights = self._compute_mean_weights(senteval_tasks)

            # 4. NLI (Usa pesos do NLI puro)
            elif name_agg == 'NLI':
                if 'NLI' in self.dynamic_weights:
                    weights = self.dynamic_weights['NLI']

            # 5. LEAVE-ONE-OUT-SENTEVAL-NLI (Média de SentEval + NLI - Task Atual)
            elif name_agg == 'LEAVE-ONE-OUT-SENTEVAL-NLI':
                target_tasks = [t for t in all_tasks_with_nli if t != self.current_task_name]
                weights = self._compute_mean_weights(target_tasks)

            # 6. AVG-ALL (Média de tudo + NLI) [Igual ao antigo UNIVERSAL]
            elif name_agg == 'AVG-ALL':
                if 'AVG_ALL' in self.dynamic_weights:
                    weights = self.dynamic_weights['AVG_ALL']
                else:
                    weights = self._compute_mean_weights(all_tasks_with_nli)

            # --- FALLBACK DE SEGURANÇA ---
            # Se não achou pesos (ex: task nova sem pesos no JSON), usa média uniforme
            if weights is None:
                # Tenta fallback para AVG_ALL se não for uma estratégia Leave-Out específica
                if 'AVG_ALL' in self.dynamic_weights:
                    weights = self.dynamic_weights['AVG_ALL']
                else:
                    weights = [1.0/len(hidden_states[-12:])] * len(hidden_states[-12:])

            # Aplicação dos Pesos
            layers_to_fuse = torch.stack(hidden_states[-12:], dim=0) 
            weights_tensor = torch.tensor(weights, device=self.device, dtype=layers_to_fuse.dtype)
            weights_tensor = weights_tensor.view(-1, 1, 1, 1) 
            fused_layer = (layers_to_fuse * weights_tensor).sum(dim=0) 
            
            return self._get_pooling_result(fused_layer, attention_mask, name_pooling, name_agg, input_ids)

        # ... (Resto do código para LYR, SUM-1-12, AVG-1-12 continua igual) ...
        if name_agg.startswith("LYR"):
            layer_idx = int(name_agg.split('-')[-1])   
            LYR_hidden =  hidden_states[layer_idx]            
            return self._get_pooling_result(LYR_hidden, attention_mask, name_pooling, "LYR", input_ids)        
        else:        
            name_agg_type = name_agg.split("-")[0]
            if "-" in name_agg:
                agg_initial_layer = int(name_agg.split("-")[1])
                agg_final_layer = int(name_agg.split("-")[2])
                
                match name_agg_type:  
                    case "SUM":
                        return self._get_pooling_result(torch.stack(hidden_states[agg_initial_layer:agg_final_layer+1], dim=0).sum(dim=0), attention_mask, name_pooling, name_agg, input_ids)
                    case "AVG":
                        return self._get_pooling_result(torch.stack(hidden_states[agg_initial_layer:agg_final_layer+1], dim=0).mean(dim=0), attention_mask, name_pooling, name_agg, input_ids)
            return None

def run_senteval(model_name, tasks, args, type_task):
    results_general = {}
    device = functions_code.get_device()
    print(f"\nExecuting Device: {device}")
    
    # PASSANDO O CAMINHO DOS PESOS DE ATENÇÃO AQUI
    encoder = SentenceEncoder(model_name, device, args.dynamic_weights_path, args.attention_weights_path)
    pooling_strategies, list_poolings, list_layers = functions_code.strategies_pooling_list(args, encoder.qtd_layers)

    print("LISTA DE POOLINGS: ", list_poolings)
    print("LISTA DE LAYERS: ", list_layers)
   
    tempos = []  
    for pooling in pooling_strategies:
        encoder.pooling_strategy = pooling
        print(f"Running: Model={encoder.name_model}, Pooling={encoder.pooling_strategy}")
        if type_task == 'cl':
            senteval_params = {
                'task_path': 'data', 'usepytorch': False, 'kfold': args.kfold,
                'classifier': {'nhid': args.nhid, 'optim': args.optim, 'batch_size': args.batch, 'tenacity': 5, 'epoch_size': args.epochs},
                'encoder': encoder
            }
        else:
             senteval_params = {'task_path': 'data', 'usepytorch': False, 'kfold': 10, 'encoder': encoder}
        
        se = senteval.engine.SE(senteval_params, functions_code.batcher)
        start_time = time.time()
        results_general[pooling] = se.eval(tasks)
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        tempos.append(elapsed_time)
        
        # ... (Logica de tempo mantida) ...
        tempo_faltante = (tempos[-1] * (len(pooling_strategies) - len(tempos))) / 60
        dias_faltante = tempo_faltante / 24
        results_general[pooling]['out_vec_size'] = encoder.size_embedding
        results_general[pooling]['qtd_layers'] = encoder.qtd_layers
        results_general[pooling]['best_layers'] = encoder.print_best_layers

        print("\nProgress: " + str(len(tempos)) + '/' + str(len(pooling_strategies)))
        print(f"--> Time for this run: {elapsed_time:.2f} minutes")     
        print(f"--> Tempo Faltante Estimado: {tempo_faltante:.2f} horas")  
        print(f"--> Dias Faltante Estimado: {dias_faltante:.2f} dias")
                              
    return results_general

# ... (Tasks Run e Main Evaluate mantidos iguais) ...
def tasks_run(args, main_path, filename_task, tasks_list, type_task):
    path_created = main_path + '/' + filename_task
    os.makedirs(path_created, exist_ok=True)
    logging.basicConfig(filename=path_created + '/' + filename_task + '_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    results_data = []
    config_path = os.path.join(path_created, f"config_{args.save_dir}.json")
    with open(config_path, 'w') as f: json.dump(vars(args), f, indent=4)

    for model_name in args.models:
        print(f"\nExecuting Model: {model_name}")
        results = run_senteval(model_name, tasks_list, args, type_task)
        for pooling, res in results.items():
            if type_task == 'cl': dict_results = [res.get(task, {}) for task in tasks_list]
            elif type_task == 'si': dict_results = [res.get(task, {}).get('all', 0) for task in tasks_list[:5]] + [res.get(task, {}) for task in tasks_list[-2:]]
            
            results_data.append({
                "model": model_name, "pooling": pooling, "out_vec_size": res.get('out_vec_size'), "best_layers": res.get('best_layers'),
                "epochs": args.epochs, "nhid": args.nhid, "qtd_layers": res.get('qtd_layers'),              
                **{task: dict_results[i] for i, task in enumerate(tasks_list)}
            })
            final_df1 = pd.DataFrame(results_data)
            final_df1.to_csv(path_created + '/' + filename_task + '_intermediate.csv', index=False)
                    
    final_df = pd.DataFrame(results_data)
    final_df.to_csv(path_created + '/' + filename_task + '.csv', index=False)
    os.remove(path_created + '/' + filename_task + '_intermediate.csv')
    functions_code.main_evaluate(final_df, type_task, path_created, filename_task, tasks_list)

def main(args):
    args.models = args.models.split(",")
    args.poolings = args.poolings.split(",")
    args.agg_layers = args.agg_layers.split(",")  
    main_path = '../results_pooling_paper_weights/' + str(args.save_dir)
    filename_task = str(args.save_dir)
    
    if args.task_type == "classification":      
        filename_cl = "cl" + filename_task
        classification_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']        
        classification_tasks = args.tasks.split(",") if args.tasks is not None else classification_tasks
        tasks_run(args, main_path, filename_cl, classification_tasks, 'cl')
    elif args.task_type == "similarity":
        filename_si = "si" + filename_task
        similarity_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        similarity_tasks = args.tasks.split(",") if args.tasks is not None else similarity_tasks
        tasks_run(args, main_path, filename_si, similarity_tasks, 'si')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentEval Experiments")
    parser.add_argument("--task_type", type=str, default='classification', choices=['classification', 'similarity'], help="Tipo de tarefa (classification ou similarity)")
    parser.add_argument("--models", type=str, required=True, help="Modelos separados por vírgula (sem espaços)")
    parser.add_argument("--epochs", type=int, default=4, help="Número máximo de épocas do classificador linear")
    parser.add_argument("--batch", type=int, default=64, help="Batch Size do classificador")
    parser.add_argument("--kfold", type=int, default=10, help="KFold para validação")
    parser.add_argument("--optim", type=str, default='adam', help="otimizador do classificador")
    parser.add_argument("--nhid", type=int, default=0, help="Numero de camadas ocultas (0 = Logistic Regression, 1 ou mais = MLP)")
    parser.add_argument("--initial_layer", default=12, type=int, help="Camada inicial para execução dos experimentos (default metade superior)")
    parser.add_argument("--final_layer", type=int, default=12, help="Camada inicial para execução dos experimentos (default metade superior)")
    parser.add_argument("--poolings", type=str, required=True, default="all", help="Poolings separados por virgula (sem espacos) ou simple, simple-ns, two, three")
    parser.add_argument("--agg_layers", type=str, required=True, default="ALL", help="agg layers separados por virgula (sem espacos)")
    parser.add_argument("--tasks", type=str, help="tasks separados por virgula (sem espacos)")
    parser.add_argument("--save_dir", type=str, help="tasks separados por virgula (sem espacos)")
    
    # NOVOS ARGUMENTOS
    parser.add_argument("--dynamic_weights_path", type=str, help="Caminho para o JSON com os pesos de Fusão (Cap 3)")
    parser.add_argument("--attention_weights_path", type=str, help="Caminho para o .pt com os pesos de Atenção (Cap 4)")

    args = parser.parse_args()
    main(args)