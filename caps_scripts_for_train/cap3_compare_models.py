import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Caminhos das pastas geradas
BASE_PATH = '/content/drive/MyDrive/Doutorado/experiments_cap3_senteval'
MODELS = {
    'BERT': 'results_cap3_bert-base-uncased',
    'RoBERTa': 'results_cap3_facebook_roberta-base',
    'DeBERTa': 'results_cap3_microsoft_deberta-v3-base'
}

data = []

for label, folder in MODELS.items():
    csv_path = os.path.join(BASE_PATH, folder, 'final_accuracies.csv')
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # O CSV tem formato "85.20 ± 0.15". Vamos limpar para pegar só o número.
        for _, row in df.iterrows():
            task = row['Task']
            acc_str = row['Acc (Mean ± Std)'].split('±')[0].strip()
            acc = float(acc_str)
            data.append({'Model': label, 'Task': task, 'Accuracy': acc})
    else:
        print(f"Ainda não encontrei resultados para {label}...")

if len(data) > 0:
    df_all = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(14, 8))
    sns.set(style="whitegrid")
    
    # Gráfico de barras agrupado
    chart = sns.barplot(data=df_all, x='Task', y='Accuracy', hue='Model', palette='viridis')
    
    plt.title('Comparação de Performance: Fusão Dinâmica em Diferentes Arquiteturas', fontsize=16)
    plt.ylim(0.5, 1.0) # Ajuste conforme suas acurácias (ex: de 50% a 100%)
    plt.legend(title='Backbone Model')
    plt.ylabel('Acurácia Média')
    
    # Salva
    plt.savefig(os.path.join(BASE_PATH, 'comparativo_final_modelos.png'), dpi=300)
    plt.show()
    print("Gráfico comparativo gerado com sucesso!")
else:
    print("Nenhum dado encontrado. Espere o treinamento terminar.")