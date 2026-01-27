import mlflow
import json
import pandas as pd
import os

# --- 1. CONFIGURAÇÕES E DEFINIÇÕES DE ARQUIVOS ---

# URI do MLflow Tracking Server (deve estar rodando no seu Ubuntu)
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000" 

# Nomes dos arquivos (assumimos que eles têm o mesmo número de linhas e representam a mesma run)
CONFIG_FILE = "/home/yansoaresdell/experimentos_yan_dell16gb/results_pooling_paper/teste_evaluate/cl_models_deberta-base_epochs_1_batch_1024_nhid_0_inlayer_12_filayer_12_pooling_CLS_agglayers_LYR-12/config_teste_evaluate.json"
DEV_ACC_FILE = "/home/yansoaresdell/experimentos_yan_dell16gb/results_pooling_paper/teste_evaluate/cl_models_deberta-base_epochs_1_batch_1024_nhid_0_inlayer_12_filayer_12_pooling_CLS_agglayers_LYR-12/cl_models_deberta-base_epochs_1_batch_1024_nhid_0_inlayer_12_filayer_12_pooling_CLS_agglayers_LYR-12_processado_devacc.csv"
ACC_FILE = "/home/yansoaresdell/experimentos_yan_dell16gb/results_pooling_paper/teste_evaluate/cl_models_deberta-base_epochs_1_batch_1024_nhid_0_inlayer_12_filayer_12_pooling_CLS_agglayers_LYR-12/cl_models_deberta-base_epochs_1_batch_1024_nhid_0_inlayer_12_filayer_12_pooling_CLS_agglayers_LYR-12_processado_acc.csv"

# Nomes das tarefas no CSV para logar as métricas
TASK_NAMES = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']

# Lista de colunas do CSV que são consideradas PARÂMETROS
# As colunas de métricas (TASK_NAMES e avg_tasks) serão ignoradas aqui e logadas como métricas.
CSV_PARAM_COLUMNS = ['model', 'type_pooling', 'agg', 'layer', 'epochs', 
                     'out_vec_size', 'qtd_layers', 'nhid', 'best_layers']

JSON_PARAMS_COLUMNS = ['batch', 'kfold', 'optim', 'nhid', 'initial_layer', 
                       'final_layer', 'save_dir']

# --- 2. FUNÇÃO PRINCIPAL DE REGISTRO NO MLFLOW ---
def log_experiment_to_mlflow():
    """Carrega dados e itera sobre as linhas do CSV, criando duas runs (DEV/ACC) para cada."""
    
    # --- 2.1. Carregar Dados ---
    try:
        df_dev = pd.read_csv(DEV_ACC_FILE)
        df_acc = pd.read_csv(ACC_FILE)
        
        # Certifica-se de que os DataFrames têm o mesmo número de runs
        if len(df_dev) != len(df_acc):
            print("AVISO: Os arquivos DEV e ACC têm números diferentes de linhas (runs).")

        with open(CONFIG_FILE, 'r') as f:
            json_params = json.load(f)

    except FileNotFoundError as e:
        print(f"ERRO: Arquivo não encontrado. Certifique-se de que '{e.filename}' está no diretório.")
        return
    except Exception as e:
        print(f"ERRO ao carregar dados: {e}")
        return


    # --- 2.2. Configurar MLflow ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Define o nome do experimento (baseado no JSON ou em um valor padrão)
    model_name_default = json_params.get('models', ['unknown'])[0]
    experiment_name = f"Classification_New1"
    mlflow.set_experiment(experiment_name)

    print(f"--- Iniciando Registro de {len(df_dev)} Run(s) ---")
    
    # Itera sobre cada linha (cada experimento/conjunto de parâmetros)
    for i in range(len(df_dev)):
        metrics_dev = df_dev.iloc[i]
        
        # Garante que a linha ACC existe
        if i < len(df_acc):
            metrics_acc = df_acc.iloc[i]
        else:
            print(f"Pulando linha ACC {i}: Não existe correspondente no arquivo ACC.")
            continue
            
        # 2.3. Configurações da Run (Baseado no CSV)
        model_name = metrics_dev.get('model', model_name_default)
        pooling_name = metrics_dev.get('pooling', 'NOPOOL')
               
        base_run_name = f"{model_name}_{pooling_name}_ID{i}"

        
        # --- 2.4. REGISTRO DAS DUAS RUNS (DEV e ACC) ---
        
        # Lista de duplas (nome do dataset, série de métricas)
        runs_to_log = [
            ("DEV", metrics_dev),
            ("ACC", metrics_acc)
        ]
        
        for dataset_name, metrics_data in runs_to_log:
            
            run_name = f"{base_run_name}_{dataset_name}"
            
            with mlflow.start_run(run_name=run_name, nested=True): # Usamos nested=True para agrupar visualmente se a UI suportar
                print(f"\n-> Registrando Run: {run_name}")
                
                # --- PARÂMETRO EXCLUSIVO DESTE NOVO REQUISITO ---
                mlflow.log_param("dataset_run", dataset_name)
                
                # --- REGISTRO DE PARÂMETROS (PRIORIDADE: CSV) ---
                
                # Loga Parâmetros do CSV
                logged_params = {}
                for col in CSV_PARAM_COLUMNS:
                    if col in metrics_data.index:
                        value = metrics_data[col]
                        mlflow.log_param(col, value)
                        logged_params[col] = value

                for col_json in JSON_PARAMS_COLUMNS:
                    valor = json_params.get(col_json, 'unknown')
                    mlflow.log_param(col_json, valor)
                
                # --- REGISTRO DE MÉTRICAS (Apenas as do dataset atual) ---
                mlflow.log_metric("avg_tasks", metrics_data['avg_tasks'])
                for task in TASK_NAMES:
                    # Loga as métricas de cada tarefa
                    mlflow.log_metric(task, metrics_data[task])                    
                
                # --- REGISTRO DE ARTEFATOS (Apenas uma vez por conjunto de parâmetros, mas aqui para ambos) ---
                # É redundante, mas garante que os arquivos estão sempre lá
                mlflow.log_artifact(CONFIG_FILE, artifact_path=f"config_{dataset_name}")
                mlflow.log_artifact(DEV_ACC_FILE, artifact_path=f"results_{dataset_name}")
                mlflow.log_artifact(ACC_FILE, artifact_path=f"results_{dataset_name}")
        
    print("\n--- REGISTRO DE TODAS AS RUNS CONCLUÍDO COM SUCESSO! ---")
    print(f"Acesse a interface: {MLFLOW_TRACKING_URI}")


# --- 3. EXECUTAR ---
if __name__ == "__main__":
    try:
        log_experiment_to_mlflow()
    except Exception as e:
        print(f"\nERRO FATAL. Verifique se o servidor está rodando em {MLFLOW_TRACKING_URI}.")
        print(f"Detalhes do Erro: {e}")