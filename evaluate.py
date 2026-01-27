import argparse
import pandas as pd
import os
import shutil

columns_tasks_cl = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']  
columns_tasks_si = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']

main_colunas = ['model', 'pooling', 'type_pooling','agg', 'layer', 'epochs', 'out_vec_size', 'qtd_layers', 'nhid', 'params', 'best_layers']

tables_processed = 'tables_processed'

ordem_colunas_cl = main_colunas + columns_tasks_cl 
ordem_colunas_si = main_colunas + columns_tasks_si 

def move_with_replace(src, dst):
    # Se o destino já existir, remova-o
    if os.path.exists(dst):
        if os.path.isdir(dst):
            shutil.rmtree(dst)  # Remove diretório
        else:
            os.remove(dst)  # Remove arquivo
    # Move o arquivo ou diretório
    shutil.move(src, dst)

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

    simple_poolings = ['CLS', 'AVG', 'SUM', 'MAX']
    simple_ns_poolings = ['AVG-NS', 'SUM-NS', 'MAX-NS']
    two_tokens_poolings = ['CLS+AVG', 'CLS+SUM', 'CLS+MAX', 'CLS+AVG-NS', 'CLS+SUM-NS', 'CLS+MAX-NS',
                           'AVG+SUM', 'AVG+MAX', 'AVG+AVG-NS', 'AVG+SUM-NS', 'AVG+MAX-NS', 
                           'SUM+MAX', 'SUM+AVG-NS', 'SUM+SUM-NS', 'SUM+MAX-NS',
                           'MAX+AVG-NS', 'MAX+SUM-NS', 'MAX+MAX-NS',
                           'AVG-NS+SUM-NS', 'AVG-NS+MAX-NS',
                           'SUM-NS+MAX-NS']
    three_tokens_poolings = ['CLS+AVG+SUM', 'CLS+AVG+MAX', 'CLS+AVG+AVG-NS', 'CLS+AVG+SUM-NS', 'CLS+AVG+MAX-NS', 
                             'CLS+SUM+MAX', 'CLS+SUM+AVG-NS', 'CLS+SUM+SUM-NS', 'CLS+SUM+MAX-NS', 'CLS+MAX+AVG-NS',
                             'CLS+MAX+SUM-NS', 'CLS+MAX+MAX-NS', 'CLS+AVG-NS+SUM-NS', 'CLS+AVG-NS+MAX-NS', 'CLS+SUM-NS+MAX-NS', 
                             'AVG+SUM+MAX', 'AVG+SUM+AVG-NS', 'AVG+SUM+SUM-NS', 'AVG+SUM+MAX-NS', 'AVG+MAX+AVG-NS', 
                             'AVG+MAX+SUM-NS', 'AVG+MAX+MAX-NS', 'AVG+AVG-NS+SUM-NS', 'AVG+AVG-NS+MAX-NS', 'AVG+SUM-NS+MAX-NS', 
                             'SUM+MAX+AVG-NS', 'SUM+MAX+SUM-NS', 'SUM+MAX+MAX-NS', 'SUM+AVG-NS+SUM-NS', 'SUM+AVG-NS+MAX-NS', 'SUM+SUM-NS+MAX-NS', 
                             'MAX+AVG-NS+SUM-NS', 'MAX+AVG-NS+MAX-NS', 'MAX+SUM-NS+MAX-NS', 
                             'AVG-NS+SUM-NS+MAX-NS']
    
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
    
def tables_process(experiment_path, cl_paths, columns_tasks_cl, ordem_colunas_cl, type_task_process):
    for clp in cl_paths:
        path_cl = os.path.join(experiment_path, clp)
        if [f for f in os.listdir(path_cl) if f.endswith('_intermediate.csv')]:
            os.remove(path_cl + '/' + [f for f in os.listdir(path_cl) if f.endswith('_intermediate.csv')][0])

        cl_file_name = [f for f in os.listdir(path_cl) if f.endswith('.csv')][0]

        caminho_arquivo_cl = os.path.join(path_cl, cl_file_name)
        data = pd.read_csv(caminho_arquivo_cl, encoding="utf-8", on_bad_lines="skip")

        devacc_data = {'model': data['model'], 'pooling': data['pooling'], 'epochs': data['epochs'], 'out_vec_size': data['out_vec_size'], 'qtd_layers': data['qtd_layers'], 'nhid': data['nhid'], 'best_layers': data['best_layers']}
        acc_data = {'model': data['model'], 'pooling': data['pooling'], 'epochs': data['epochs'], 'out_vec_size': data['out_vec_size'], 'qtd_layers': data['qtd_layers'], 'nhid': data['nhid'], 'best_layers': data['best_layers']}

        if type_task_process == 'cl':

            for task in columns_tasks_cl:
                devacc_data[task] = data[task].apply(lambda x: parse_dict_with_eval(x).get('devacc', None))
                acc_data[task] = data[task].apply(lambda x: parse_dict_with_eval(x).get('acc', None))

        elif type_task_process == 'si':

            for task in columns_tasks_cl:
                if task in columns_tasks_cl[:5]:
                    devacc_data[task] = data[task].apply(lambda x: (parse_dict_with_eval(x).get('pearson', None).get('mean', None)) * 100)
                    acc_data[task] = data[task].apply(lambda x: (parse_dict_with_eval(x).get('spearman', None).get('mean', None)) * 100)
                if task in columns_tasks_cl[5:]:
                    devacc_data[task] = data[task].apply(lambda x: (parse_dict_with_eval_other(x).get('pearson', None)) * 100)
                    acc_data[task] = data[task].apply(lambda x: (parse_dict_with_eval_other(x).get('spearman', None)) * 100)


        devacc_table = pd.DataFrame(devacc_data)
        acc_table = pd.DataFrame(acc_data)        

        devacc_table[['agg', 'layer']] = devacc_table['pooling'].str.split('_', expand=True)
        acc_table[['agg', 'layer']] = acc_table['pooling'].str.split('_', expand=True)

        devacc_table['params'] = "-".join(cl_file_name.split('_')[5:11])
        acc_table['params'] = "-".join(cl_file_name.split('_')[5:11])

        devacc_table['type_pooling'] = devacc_table['agg'].apply(get_type_pooling)
        acc_table['type_pooling'] = acc_table['agg'].apply(get_type_pooling)

        devacc_table = devacc_table[ordem_colunas_cl]
        acc_table = acc_table[ordem_colunas_cl]  

        devacc_table['avg_tasks'] = devacc_table[columns_tasks_cl].mean(axis=1)
        acc_table['avg_tasks'] = acc_table[columns_tasks_cl].mean(axis=1)

        if type_task_process == 'cl':
            devacc_table.to_csv(os.path.join(path_cl, cl_file_name.split('.csv')[0] + '_processado_devacc.csv'))
            acc_table.to_csv(os.path.join(path_cl, cl_file_name.split('.csv')[0]) + '_processado_acc.csv')

        elif type_task_process == 'si':
            devacc_table.to_csv(os.path.join(path_cl, cl_file_name.split('.csv')[0] + '_processado_pearson.csv'))
            acc_table.to_csv(os.path.join(path_cl, cl_file_name.split('.csv')[0]) + '_processado_spearman.csv')

        #devacc_table.to_csv(os.path.join(tables_processed, cl_file_name.split('.csv')[0] + '_processado_devacc.csv'))
        #acc_table.to_csv(os.path.join(tables_processed, cl_file_name.split('.csv')[0]) + '_processado_acc.csv')

def main(experiment_path):
    if args.task_type == "classification":
        cl_paths = [p for p in os.listdir(experiment_path) if p.startswith('cl_')]
        tables_process(experiment_path, cl_paths, columns_tasks_cl, ordem_colunas_cl, 'cl')

    elif args.task_type == "similarity":
        si_paths = [p for p in os.listdir(experiment_path) if p.startswith('si_')]
        tables_process(experiment_path, si_paths, columns_tasks_si, ordem_colunas_si, 'si')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Experiments")
    parser.add_argument("--task_type", type=str, required=True, default="classification", help="Tipo de tarefa (classification ou similarity)")
    parser.add_argument("--save_dir", required=True, type=str, help="Diretório do experimento que contém os modelos.")
    
    args = parser.parse_args()

    #experiment_path = '../results_pooling_paper/' + str(args.save_dir)
    experiment_path = str(args.save_dir)
    main(experiment_path)