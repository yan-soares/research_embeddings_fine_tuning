import json
import numpy as np

# Entrada: Seu arquivo com os pesos aprendidos (Specifics)
INPUT_FILE = 'dynamic_weights_roberta-base.json' 
OUTPUT_FILE = 'dynamic_weights_roberta-base_honest.json'

def main():
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    # Pega todas as tasks que têm pesos aprendidos (excluindo chaves especiais)
    tasks = [k for k in data.keys() if k not in ['AVG_ALL', 'SUM_ALL'] and isinstance(data[k], list)]
    
    honest_data = {}
    
    print("Gerando 'Universal Honesto' (Leave-One-Out)...")
    
    for target_task in tasks:
        # Pega todas as outras tasks para fazer a média
        source_tasks = [t for t in tasks if t != target_task]
        
        # Coleta os vetores
        vectors = [data[src] for src in source_tasks]
        
        if vectors:
            # Calcula a média (Simula o AVG_ALL sem a task alvo)
            avg_vector = np.mean(vectors, axis=0).tolist()
            
            # Salva com o NOME da task alvo. 
            # Isso é o "pulo do gato" para seu main.py usar.
            honest_data[target_task] = avg_vector
            print(f" -> Para avaliar {target_task}, usaremos a média de: {source_tasks}")

    # Salva
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(honest_data, f, indent=4)
    
    print(f"\nArquivo '{OUTPUT_FILE}' salvo.")

if __name__ == "__main__":
    main()