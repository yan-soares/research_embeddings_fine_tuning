import torch
from itertools import combinations
import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModel, DebertaV2Model, DebertaV2Tokenizer, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel

main_colunas = ['model', 'pooling', 'type_pooling','agg', 'layer', 'epochs', 'out_vec_size', 'qtd_layers', 'nhid', 'params', 'best_layers']

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

    devacc_table['params'] = "-".join(filename_task.split('_')[5:11])
    acc_table['params'] = "-".join(filename_task.split('_')[5:11])

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
