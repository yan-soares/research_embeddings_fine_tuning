import torch
from itertools import combinations
import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModel, DebertaV2Model, DebertaV2Tokenizer, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, AutoModelForMaskedLM
import torch
import os
import json
from nltk.corpus import stopwords
import numpy as np

main_colunas = ['model', 'pooling', 'type_pooling','agg', 'layer', 'epochs', 'out_vec_size', 'qtd_layers', 'nhid', 'params', 'best_layers']

class SentenceEncoder:
    def __init__(self, model_name, device, args):
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

        self.run_pooling = None
        self.run_layer = None

        # --- Inicializa o modelo escolhido ---
        self.name_model, self.qtd_layers, self.tokenizer, self.model = load_model(model_name, device)
        self._prepare_special_token_ids()

        # --- FUSÃO (CAP 5) ---
        self.dynamic_weights = {}
        self.dynamic_weights_path = args.dynamic_weights_path
        if self.dynamic_weights_path and os.path.exists(self.dynamic_weights_path):
            print(f"Carregando Pesos de Fusão do Cap 5 de: {self.dynamic_weights_path}")
            with open(self.dynamic_weights_path, 'r') as f:
                self.dynamic_weights = json.load(f)
        else:
            if self.dynamic_weights_path:
                print(f"AVISO: Arquivo de fusão {self.dynamic_weights_path} não encontrado.")
       
    def _prepare_special_token_ids(self):
        stopwords_list = stopwords.words('english')
        stopword_ids = self.tokenizer.convert_tokens_to_ids(stopwords_list)
        stopword_ids_filtered = [id for id in stopword_ids if id != self.tokenizer.unk_token_id]
        self.stopwords_set_ids = torch.tensor(stopword_ids_filtered, device=self.device)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

    def _encode(self, sentences, current_task, batch_size=2048): 
        self.current_task_name = current_task
        tokens = self.tokenizer(sentences, padding="longest", truncation=True, return_tensors="pt", max_length=self.model.config.max_position_embeddings)
        tokens = {key: val.to(self.device, non_blocking=True) for key, val in tokens.items()}
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch_tokens = {key: val[i:i+batch_size] for key, val in tokens.items()}
            use_amp = (self.device.type == "cuda")
            with torch.inference_mode(), torch.amp.autocast(device_type="cuda", enabled=use_amp):
                output = self.model(**batch_tokens, output_hidden_states=True, return_dict=True)
                embeddings = self._apply_pooling(output, batch_tokens['attention_mask'], batch_tokens['input_ids'])
            all_embeddings.append(embeddings)

        self.size_embedding = all_embeddings[0].shape[-1]
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
        return sum_embeddings / valid_token_counts.clamp(min=1e-4)

    def _sum_pooling_exclude_cls_sep_and_stopwords(self, output, attention_mask, input_ids):
        mask = self._create_combined_mask(input_ids, attention_mask, exclude_stopwords=True, exclude_cls_sep=True)
        expanded_mask = mask.unsqueeze(-1)
        return (output * expanded_mask).sum(dim=1)
    
    def _max_pooling_exclude_cls_sep_and_stopwords(self, output, attention_mask, input_ids):
        mask = self._create_combined_mask(input_ids, attention_mask, exclude_stopwords=True, exclude_cls_sep=True)
        expanded_mask = mask.unsqueeze(-1)
        masked_embeddings = output.masked_fill(expanded_mask == 0, -1e4)
        return masked_embeddings.max(dim=1)[0]
    
    def _simple_pooling(self, hidden_state, attention_mask, name_pooling, input_ids):
        match name_pooling:
            case "CLS":
                self.run_pooling = "CLS"
                return hidden_state[:, 0, :]
            case "AVG":
                self.run_pooling = "AVG"
                return ((hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-4))
            case "SUM":
                self.run_pooling = "SUM"
                return (hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1)
            case "MAX":
                self.run_pooling = "MAX"
                return torch.max(hidden_state.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e4), dim=1)[0]
            case "AVG-NS":
                self.run_pooling = "AVG-NS"
                return self._mean_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)
            case "SUM-NS":
                self.run_pooling = "SUM-NS"
                return self._sum_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)
            case "MAX-NS":
                self.run_pooling = "MAX-NS"
                return self._max_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)
                         
    def _get_pooling_result(self, hidden_state, attention_mask, name_pooling, name_agg, input_ids):
        self.print_best_layers =  "NORMAL"        
        name_pooling_split = name_pooling.split('+')

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

    def _apply_pooling(self, output, attention_mask, input_ids):  
        hidden_states = output.hidden_states
        name_pooling = self.pooling_strategy.split("_")[0] 
        name_agg = self.pooling_strategy.split("_")[-1]    

        senteval_tasks_cl = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
        senteval_nli = senteval_tasks_cl + ['NLI']

        dynamic_weights_strategies = [
            'IN-TASK', 'LOO-SENTEVAL', 'AVG-SENTEVAL', 
            'NLI', 'LOO-SENTEVAL-NLI', 'AVG-ALL', 'STSB'
        ]

        if name_agg in dynamic_weights_strategies:
            weights = None
            
            # 1. IN-TASK (Usa os pesos da própria tarefa atual) [Igual ao antigo DYNAMIC]
            if name_agg == 'IN-TASK':
                if self.current_task_name in self.dynamic_weights:
                    weights = self.dynamic_weights[self.current_task_name]
                    self.run_layer = "IN-TASK"

            # 2. LEAVE-ONE-OUT-SENTEVAL (Média do SentEval - Task Atual)
            elif name_agg == 'LOO-SENTEVAL':
                target_tasks = [t for t in senteval_tasks_cl if t != self.current_task_name]
                weights = self._compute_mean_weights(target_tasks)
                self.run_layer = "LOO-SENTEVAL"

            # 3. AVG-SENTEVAL (Média de todas do SentEval)
            elif name_agg == 'AVG-SENTEVAL':
                # Tenta pegar direto do JSON se existir, senão calcula
                if 'AVG_SENTEVAL' in self.dynamic_weights:
                    weights = self.dynamic_weights['AVG_SENTEVAL']
                    self.run_layer = "AVG-SENTEVAL"
                else:
                    weights = self._compute_mean_weights(senteval_tasks_cl)
                    self.run_layer = "AVG-SENTEVAL"

            # 4. NLI (Usa pesos do NLI puro)
            elif name_agg == 'NLI':
                if 'NLI' in self.dynamic_weights:
                    weights = self.dynamic_weights['NLI']
                    self.run_layer = "NLI"

            # 5. LEAVE-ONE-OUT-SENTEVAL-NLI (Média de SentEval + NLI - Task Atual)
            elif name_agg == 'LOO-SENTEVAL-NLI':
                target_tasks = [t for t in senteval_nli if t != self.current_task_name]
                weights = self._compute_mean_weights(target_tasks)
                self.run_layer = "LOO-SENTEVAL-NLI"

            # 6. AVG-ALL (Média de tudo + NLI) [Igual ao antigo UNIVERSAL]
            elif name_agg == 'AVG-ALL':
                if 'AVG_ALL' in self.dynamic_weights:
                    weights = self.dynamic_weights['AVG_ALL']
                    self.run_layer = "AVG-ALL"
                else:
                    weights = self._compute_mean_weights(senteval_nli)
                    self.run_layer = "AVG-ALL-CALCULADO"

            # 7. STS: Usar os pesos do STSB, para tarefas de similaridade
            elif name_agg == 'STSB':
                # tenta chave universal
                if 'STSBenchmark' in self.dynamic_weights:
                    weights = self.dynamic_weights['STSBenchmark']
                    self.run_layer = "STSB"
                else:
                    # média de todas as tasks de similaridade que existirem no JSON
                    sts_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
                    weights = self._compute_mean_weights(sts_tasks)
                    self.run_layer = "STS-MEAN"

            # --- FALLBACK DE SEGURANÇA ---
            # Se não achou pesos (ex: task nova sem pesos no JSON), usa média uniforme
            if weights is None:
                # Tenta fallback para AVG_ALL se não for uma estratégia Leave-Out específica
                if 'AVG_ALL' in self.dynamic_weights:
                    weights = self.dynamic_weights['AVG_ALL']
                    self.run_layer = "AVG-ALL-ELSE"
                else:
                    weights = [1.0/len(hidden_states[-12:])] * len(hidden_states[-12:])
                    self.run_layer = "AVG-ALL-UNIFORME"

            encoder_layers = hidden_states[1:]
            L = len(encoder_layers)

            if weights is None:
                print("sem pesos")
            else:
                if len(weights) != L:
                    raise ValueError(
                        f"[Layer mismatch] task={self.current_task_name} agg={name_agg} "
                        f"len(weights)={len(weights)} len(layers)={L}"
                    )

            layers_to_fuse = torch.stack(encoder_layers, dim=0)
            weights_tensor = torch.tensor(weights, device=layers_to_fuse.device, dtype=layers_to_fuse.dtype)
            weights_tensor = weights_tensor.view(-1, 1, 1, 1)
            fused_layer = (layers_to_fuse * weights_tensor).sum(dim=0)
            
            return self._get_pooling_result(fused_layer, attention_mask, name_pooling, name_agg, input_ids)

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

def load_model(model_name, device):

    name_model = None
    qtd_layers = None
    tokenizer = None
    model = None

    if model_name == 'bert-base' or model_name == 'bert-large':
        if model_name == 'bert-base':
            name_model = 'google-bert/bert-base-uncased'
            qtd_layers = 12
        if model_name == 'bert-large':
            name_model = 'google-bert/bert-large-uncased'
            qtd_layers = 24
        tokenizer = BertTokenizer.from_pretrained(name_model)
        try:
            model = BertModel.from_pretrained(name_model, output_hidden_states=True, attn_implementation="flash_attention_2").to(device)
        except:
            model = BertModel.from_pretrained(name_model, output_hidden_states=True).to(device)

    elif model_name == 'roberta-base' or  model_name == 'roberta-large':
        if model_name == 'roberta-base':
            name_model = 'FacebookAI/roberta-base'
            qtd_layers = 12
        if model_name == 'roberta-large':
            name_model = 'FacebookAI/roberta-large'
            qtd_layers = 24
        tokenizer = RobertaTokenizer.from_pretrained(name_model)
        try:
            model = RobertaModel.from_pretrained(name_model, output_hidden_states=True, attn_implementation="flash_attention_2").to(device)
        except:
            model = RobertaModel.from_pretrained(name_model, output_hidden_states=True).to(device)

    elif model_name == 'deberta-base' or model_name == 'deberta-large':
        if model_name == 'deberta-base':
            name_model = 'microsoft/deberta-v3-base'
            qtd_layers = 12
        if model_name == 'deberta-large':
            name_model = 'microsoft/deberta-v3-large'
            qtd_layers = 24
        tokenizer = DebertaV2Tokenizer.from_pretrained(name_model)
        try:
            model = DebertaV2Model.from_pretrained(name_model, output_hidden_states=True, attn_implementation="flash_attention_2").to(device)
        except:
            model = DebertaV2Model.from_pretrained(name_model, output_hidden_states=True).to(device)

    elif model_name == 'modern-bert-base' or model_name == 'modern-bert-large':
        if model_name == 'modern-bert-base':
            name_model = 'answerdotai/ModernBERT-base'
            qtd_layers = 22
        if model_name == 'modern-bert-large':
            name_model = 'answerdotai/ModernBERT-large'
            qtd_layers = 28
        tokenizer = AutoTokenizer.from_pretrained(name_model)
        try:
            model = AutoModel.from_pretrained(name_model, output_hidden_states=True, attn_implementation="flash_attention_2").to(device)
        except:
            model = AutoModel.from_pretrained(name_model, output_hidden_states=True).to(device)
    
    elif model_name == 'allmpnet':
        name_model = 'sentence-transformers/all-mpnet-base-v2'
        qtd_layers = 12
        tokenizer = AutoTokenizer.from_pretrained(name_model)
        try:
            model = AutoModel.from_pretrained(name_model, output_hidden_states=True, attn_implementation="flash_attention_2").to(device)
        except:
            model = AutoModel.from_pretrained(name_model, output_hidden_states=True).to(device)

    return name_model, qtd_layers, tokenizer, model

def get_agg_list(type_agg, range_i, range_f):

    #BASE range de 1 a 13
    #LARGE range de 1 a 25

    list_agg = []

    ranges = list(range(range_i, range_f))

    slices = {}
    for size in range(range_i + 1, range_f):  # De 2 até 12 ou de 2 até 24
        slices[size] = [f"{type_agg}-{group[0]}-{group[-1]}" for group in (ranges[i:i+size] for i in range(len(ranges) - size + 1))]

    for size, groups in slices.items():
        list_agg+=groups

    return list_agg

def get_pooling_techniques(poolings_args):

    simple_ns_poolings = ['AVG-NS', 'SUM-NS', 'MAX-NS'] 
    simple_poolings = ['CLS', 'AVG', 'SUM', 'MAX']   

    all_poolings_individuals = simple_poolings + simple_ns_poolings ## 7
    two_tokens_poolings = [f"{a}+{b}" for a, b in combinations(all_poolings_individuals, 2)] ## 21
    three_tokens_poolings = [f"{a}+{b}+{c}" for a, b, c in combinations(all_poolings_individuals, 3)] ##35

    pooling_prefixs = []
    
    if poolings_args[0] == 'all':
        pooling_prefixs = all_poolings_individuals + two_tokens_poolings + three_tokens_poolings
        return pooling_prefixs    
    
    if 'simple' in poolings_args:
        pooling_prefixs += simple_poolings
        #return pooling_prefixs
    if 'simple_all' in poolings_args:
        pooling_prefixs += all_poolings_individuals
        #return pooling_prefixs
    if 'simple-ns' in poolings_args:
        pooling_prefixs += simple_ns_poolings
        #return pooling_prefixs
    if 'two' in poolings_args:
        pooling_prefixs += two_tokens_poolings
        #return pooling_prefixs
    if 'three' in poolings_args:
        pooling_prefixs += three_tokens_poolings
        #return pooling_prefixs  
    if 'ATTENTION' in poolings_args:
        pooling_prefixs += ['ATTENTION']  
    
    if len(pooling_prefixs) > 0:
        return pooling_prefixs
    else:
        return poolings_args

def get_list_layers(final_layer, initial_layer, agg_layers_args, qtd_layers):

    list_lyrs_agg_sum = get_agg_list('SUM', 1, qtd_layers+1)
    list_lyrs_agg_avg = get_agg_list('AVG', 1, qtd_layers+1)
    list_lyrs_agg = list_lyrs_agg_sum + list_lyrs_agg_avg

    lyrs = []
        
    if agg_layers_args[0] == 'ALL':
        for i in range(initial_layer, final_layer):
            lyrs.append(f"LYR-{i+1}")
        lyrs += list_lyrs_agg
        return lyrs
    
    if agg_layers_args[0] == 'SUMAGGLAYERS':
        return list_lyrs_agg_sum    
    if agg_layers_args[0] == 'AVGAGGLAYERS':
        return list_lyrs_agg_avg 
    if agg_layers_args[0] == 'LYR':
        for i in range(initial_layer, final_layer):
            lyrs.append(f"LYR-{i+1}")
        return lyrs    
    else:
        return agg_layers_args

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def batcher(params, batch):
    # Mantém o registro dos índices originais
    original_indices = np.arange(len(batch))
    
    # Calcula o comprimento de cada sentença
    lengths = np.array([len(sent) for sent in batch])
    
    # Obtém os índices que ordenariam o batch por comprimento
    sorted_indices = np.argsort(lengths)
    
    # Ordena o batch e os índices originais
    sorted_batch = [batch[i] for i in sorted_indices]
    original_indices_sorted = [original_indices[i] for i in sorted_indices]

    # Converte para strings e chama o _encode com o batch ordenado
    sentences = [' '.join(sent) for sent in sorted_batch]
    embeddings = params['encoder']._encode(sentences, params.current_task)
    
    # Cria um array para os embeddings na ordem correta
    restored_order_embeddings = np.zeros_like(embeddings)
    
    # Usa os índices ordenados para colocar cada embedding de volta em sua posição original
    restored_order_embeddings[original_indices_sorted] = embeddings
    
    return restored_order_embeddings

def strategies_pooling_list (args, qtd_layers):
        initial_layer_args = args.initial_layer
        final_layer_args = args.final_layer
        poolings_args = args.poolings
        agg_layers_args = args.agg_layers
        
        #POOLING
        pooling_techniques = get_pooling_techniques(poolings_args)
        
        #LAYERS
        if initial_layer_args is not None:
            initial_layer = initial_layer_args
        else:
            initial_layer = int(qtd_layers / 2)

        if final_layer_args is not None:
            final_layer = final_layer_args
        else:
            final_layer = int(qtd_layers)

        list_lyrs = get_list_layers(final_layer, initial_layer, agg_layers_args, qtd_layers)

        #STRATEGIES CONCAT
        pooling_strategies = []
        for l in list_lyrs:
            for p in pooling_techniques:
                pooling_strategies.append(p + "_" + l) 

        #RETURN
        return pooling_strategies, pooling_techniques, list_lyrs

def parse_dict_with_eval(value):
    try:
        if isinstance(value, str):
            value = value.replace('np.float64', 'float')
            return eval(value)
        return {}
    except Exception as e:
        return {}

def parse_dict_with_eval_other(value):
    try:
        if isinstance(value, str):
            value = value.replace('np.float64', 'float')
            value = ','.join(value.split(',')[:3]) + '}'
            return eval(value)
        return {}
    except Exception as e:
        return {}
    
def get_type_pooling(pooling_str):

    simple_ns_poolings = ['AVG-NS', 'SUM-NS', 'MAX-NS'] 
    simple_poolings = ['CLS', 'AVG', 'SUM', 'MAX']   

    all_poolings_individuals = simple_poolings + simple_ns_poolings

    two_tokens_poolings = [f"{a}+{b}" for a, b in combinations(all_poolings_individuals, 2)]
    three_tokens_poolings = [f"{a}+{b}+{c}" for a, b, c in combinations(all_poolings_individuals, 3)]
    
    if pooling_str in simple_poolings:
        return "simple"
    if pooling_str in simple_ns_poolings:
        return "simple-ns"
    if pooling_str in two_tokens_poolings:
        return "two-tokens"
    if pooling_str in three_tokens_poolings:
        return "three-tokens"
    else:
        return "not categorized"
    
def tables_process(data, columns_tasks, type_task, path_cl, filename_task):

    ordem_colunas = main_colunas + columns_tasks

    devacc_data = {'model': data['model'], 'pooling': data['pooling'], 'epochs': data['epochs'], 'out_vec_size': data['out_vec_size'], 'qtd_layers': data['qtd_layers'], 'nhid': data['nhid'], 'best_layers': data['best_layers']}
    acc_data = {'model': data['model'], 'pooling': data['pooling'], 'epochs': data['epochs'], 'out_vec_size': data['out_vec_size'], 'qtd_layers': data['qtd_layers'], 'nhid': data['nhid'], 'best_layers': data['best_layers']}

    if type_task == 'cl':
        for task in columns_tasks:
            devacc_data[task] = data[task].apply(lambda x:x.get('devacc', None))
            acc_data[task] = data[task].apply(lambda x: x.get('acc', None))

    elif type_task == 'si':
        for task in columns_tasks:
            if task in columns_tasks[:5]:
                devacc_data[task] = data[task].apply(lambda x: (x.get('pearson', None).get('mean', None)) * 100)
                acc_data[task] = data[task].apply(lambda x: (x.get('spearman', None).get('mean', None)) * 100)
            if task in columns_tasks[5:]:
                devacc_data[task] = data[task].apply(lambda x: (x.get('pearson', None)) * 100)
                acc_data[task] = data[task].apply(lambda x: (x.get('spearman', None)) * 100)


    devacc_table = pd.DataFrame(devacc_data)
    acc_table = pd.DataFrame(acc_data)        

    devacc_table[['agg', 'layer']] = devacc_table['pooling'].str.split('_', expand=True)
    acc_table[['agg', 'layer']] = acc_table['pooling'].str.split('_', expand=True)

    devacc_table['params'] = "None"
    acc_table['params'] = "None"

    devacc_table['type_pooling'] = devacc_table['agg'].apply(get_type_pooling)
    acc_table['type_pooling'] = acc_table['agg'].apply(get_type_pooling)

    devacc_table = devacc_table[ordem_colunas]
    acc_table = acc_table[ordem_colunas]  

    devacc_table['avg_tasks'] = devacc_table[columns_tasks].mean(axis=1)
    acc_table['avg_tasks'] = acc_table[columns_tasks].mean(axis=1)

    if type_task == 'cl':
        devacc_table.to_csv(os.path.join(path_cl, filename_task + '_processado_devacc.csv'))
        acc_table.to_csv(os.path.join(path_cl, filename_task + '_processado_acc.csv'))

    elif type_task == 'si':
        devacc_table.to_csv(os.path.join(path_cl, filename_task + '_processado_pearson.csv'))
        acc_table.to_csv(os.path.join(path_cl, filename_task + '_processado_spearman.csv'))

def main_evaluate(final_df, type_task, path_for_save, filename_task, tasks_list):
    if type_task == "cl":
        tables_process(final_df, tasks_list, type_task, path_for_save, filename_task)

    elif type_task == "si":
        tables_process(final_df, tasks_list, type_task, path_for_save, filename_task)
