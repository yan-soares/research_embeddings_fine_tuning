import senteval
import pandas as pd
import torch
import argparse
import logging
import os
import json
import time
import warnings
from sklearn.exceptions import ConvergenceWarning
import functions_code
from functions_code import SentenceEncoder
from pathlib import Path

warnings.filterwarnings("ignore", category=ConvergenceWarning)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def run_senteval(model_name, tasks, args, type_task):
    results_general = {}
    device = functions_code.get_device()
    print(f"\nExecuting Device: {device}")

    encoder = SentenceEncoder(model_name, device, args)
    pooling_strategies, list_poolings, list_layers = functions_code.strategies_pooling_list(args, encoder.qtd_layers)

    print("LISTA DE POOLINGS: ", list_poolings)
    print("LISTA DE LAYERS: ", list_layers)
   
    tempos = []  
    print("-" * 60)
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
             senteval_params = {'task_path': 'data', 'usepytorch': False, 'encoder': encoder, 'batch_size': args.batch}
        
        se = senteval.engine.SE(senteval_params, functions_code.batcher)
        start_time = time.time()
        results_general[pooling] = se.eval(tasks)
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        tempos.append(elapsed_time)

        print(f"Used: Pooling={encoder.run_pooling} in Layer_Weights={encoder.run_layer}")

        tempo_faltante = (tempos[-1] * (len(pooling_strategies) - len(tempos))) / 60
        dias_faltante = tempo_faltante / 24
        results_general[pooling]['out_vec_size'] = encoder.size_embedding
        results_general[pooling]['qtd_layers'] = encoder.qtd_layers
        results_general[pooling]['best_layers'] = encoder.print_best_layers

        print("\nProgress: " + str(len(tempos)) + '/' + str(len(pooling_strategies)))
        print(f"--> Time for this run: {elapsed_time:.2f} minutes")     
        print(f"--> Tempo Faltante Estimado: {tempo_faltante:.2f} horas")  
        print(f"--> Dias Faltante Estimado: {dias_faltante:.2f} dias")
        print("-" * 60)
                              
    return results_general

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
                "model": model_name, "pooling": pooling, "out_vec_size": res.get('out_vec_size'), "best_layers": "-".join(str(Path(args.dynamic_weights_path).parent).split('/')[-2:]),
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

    if args.dynamic_weights_path and os.path.exists(args.dynamic_weights_path):
        main_path = str(Path(args.dynamic_weights_path).parent) + "/results"
        filename_task = "results"
    else:
        main_path = '../results_pooling_paper_weights/' + str(args.save_dir)
        filename_task = str(args.save_dir)
    
    if args.task_type == "classification":      
        filename_cl = "cl_" + filename_task + "_" + "-".join(str(Path(args.dynamic_weights_path).parent).split('/')[-2:])
        classification_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']        
        classification_tasks = args.tasks.split(",") if args.tasks is not None else classification_tasks
        tasks_run(args, main_path, filename_cl, classification_tasks, 'cl')
    elif args.task_type == "similarity":
        filename_si = "si_" + filename_task
        similarity_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        similarity_tasks = args.tasks.split(",") if args.tasks is not None else similarity_tasks
        tasks_run(args, main_path, filename_si, similarity_tasks, 'si')

    elif args.task_type == "glue":
        filename_glue = "glue_" + filename_task
        glue_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        glue_tasks = args.tasks.split(",") if args.tasks is not None else glue_tasks
        tasks_run(args, main_path, filename_glue, glue_tasks, 'glue')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentEval Experiments")
    parser.add_argument("--task_type", type=str, default='classification', choices=['classification', 'similarity', 'glue'], help="Tipo de tarefa (classification ou similarity)")
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
    parser.add_argument("--dynamic_weights_path", type=str, help="Caminho para o JSON com os pesos de Fusão (Cap 5)")

    args = parser.parse_args()
    main(args)