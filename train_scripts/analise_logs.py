import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import argparse

# Script para ler o JSON gerado pelo treino e plotar gráficos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Aponte este caminho para a pasta onde o JSON foi criado!
    parser.add_argument('--log_dir', type=str, required=True, help='Caminho da pasta onde está o detailed_training_logs.json')
    args = parser.parse_args()
    
    json_path = os.path.join(args.log_dir, "detailed_training_logs.json")
    
    if not os.path.exists(json_path):
        print(f"❌ Arquivo não encontrado: {json_path}")
        exit()

    print(f"📊 Lendo logs de: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    loss_rows = []
    acc_rows = []
    
    for entry in data:
        task = entry['task']
        seed = str(entry['seed'])
        history = entry['history']
        
        for epoch, val in enumerate(history['train_loss']):
            loss_rows.append({'Task': task, 'Seed': seed, 'Epoch': epoch+1, 'Loss': val})
        for epoch, val in enumerate(history['val_acc']):
            acc_rows.append({'Task': task, 'Seed': seed, 'Epoch': epoch+1, 'Accuracy': val})
            
    df_loss = pd.DataFrame(loss_rows)
    df_acc = pd.DataFrame(acc_rows)
    
    tasks = df_loss['Task'].unique()
    sns.set_theme(style="whitegrid")
    
    for task in tasks:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Loss
        task_loss = df_loss[df_loss['Task'] == task]
        sns.lineplot(data=task_loss, x='Epoch', y='Loss', hue='Seed', marker='o', ax=axes[0])
        axes[0].set_title(f"{task} - Training Loss")
        
        # Accuracy
        task_acc = df_acc[df_acc['Task'] == task]
        sns.lineplot(data=task_acc, x='Epoch', y='Accuracy', hue='Seed', marker='o', ax=axes[1])
        axes[1].set_title(f"{task} - Validation Accuracy")
        
        save_file = os.path.join(args.log_dir, f"chart_{task}.png")
        plt.tight_layout()
        plt.savefig(save_file)
        plt.close()
        print(f"✅ Gráfico salvo: {save_file}")