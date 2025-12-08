import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, accuracy_score
from utils import (limpeza_v1_simples, limpeza_v2_negacao, limpeza_v3_stemming, 
                   treinar_modelo, classificar_texto)


OUTPUT_DIR = 'assets'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"📂 Pasta '{OUTPUT_DIR}' criada para salvar os gráficos!")

def carregar_dados():
    try:
        print("📂 Carregando dataset...")
        df = pd.read_csv('data/base-reviews-b2w.csv')
        df = df[['overall_rating', 'review_title', 'review_text']].copy()
        df['review_title'] = df['review_title'].fillna('')
        df = df.dropna(subset=['review_text'])
        df['texto_completo'] = df['review_title'] + " " + df['review_text']

        def definir_sentimento(nota):
            if nota >= 4: return 'Positivo'
            elif nota == 3: return 'Neutro'
            else: return 'Negativo'

        df['sentimento'] = df['overall_rating'].apply(definir_sentimento)
        return df
    except FileNotFoundError:
        print("❌ Erro: Arquivo 'data/base-reviews-b2w.csv' não encontrado.")
        return None

def balancear_dados_treino(df_in):
    contagem = df_in['sentimento'].value_counts()
    menor_tamanho = contagem.min()
    print(f"⚖️ Balanceando classes para {menor_tamanho} exemplos cada...")
    
    df_pos = df_in[df_in['sentimento'] == 'Positivo'].sample(n=menor_tamanho, random_state=42)
    df_neg = df_in[df_in['sentimento'] == 'Negativo'].sample(n=menor_tamanho, random_state=42)
    df_neu = df_in[df_in['sentimento'] == 'Neutro'].sample(n=menor_tamanho, random_state=42)
    
    df_novo = pd.concat([df_pos, df_neg, df_neu])
    return df_novo.sample(frac=1, random_state=42).reset_index(drop=True)

def salvar_matriz(y_real, y_pred, titulo, nome_arquivo, cmap='Blues'):
    """
    Gera e salva a Matriz de Confusão.
    """

    matriz = pd.crosstab(y_real, y_pred, rownames=['Real'], colnames=['Predito'])
    

    plt.figure(figsize=(8, 6))
    
  
    sns.heatmap(matriz, annot=True, fmt='d', cmap=cmap, cbar=True)
    
    plt.title(titulo)
    plt.tight_layout()
    
   
    caminho_completo = os.path.join(OUTPUT_DIR, nome_arquivo)
    plt.savefig(caminho_completo)
    plt.close() 
    print(f"   💾 Matriz salva em: {caminho_completo}")

def executar_analise():
    df = carregar_dados()
    if df is None: return

    # --- SETUP GERAL ---
    df_embaralhado = df.sample(frac=1, random_state=42).reset_index(drop=True)
    corte = int(len(df_embaralhado) * 0.8)
    
    resultados = {} 
    acc_baseline = 0 

    # --- V1: BASELINE ---
    print("\n" + "="*50)
    print("🚀 Executando V1 (Baseline)...")
    df_v1 = df_embaralhado.copy()
    df_v1['tokens'] = df_v1['texto_completo'].apply(limpeza_v1_simples)
    
    treino_v1 = df_v1.iloc[:corte]
    teste_v1 = df_v1.iloc[corte:].copy()
    
    modelo_v1 = treinar_modelo(treino_v1)
    teste_v1['predicao'] = teste_v1['tokens'].apply(lambda x: classificar_texto(x, modelo_v1))
    
    acc_v1 = accuracy_score(teste_v1['sentimento'], teste_v1['predicao'])
    acc_baseline = acc_v1
    resultados['V1'] = classification_report(teste_v1['sentimento'], teste_v1['predicao'], output_dict=True)
    
    print(f"📊 Acurácia V1: {acc_v1:.2%}")
    salvar_matriz(teste_v1['sentimento'], teste_v1['predicao'], 
                  f"Matriz V1 (Baseline) - Acc: {acc_v1:.2%}", 
                  "matriz_v1.png", cmap='Blues')

    # --- V2: NEGAÇÃO ---
    print("\n" + "="*50)
    print("🚀 Executando V2 (Negação)...")
    df_v2 = df_embaralhado.copy()
    df_v2['tokens'] = df_v2['texto_completo'].apply(limpeza_v2_negacao)
    
    treino_v2 = df_v2.iloc[:corte]
    teste_v2 = df_v2.iloc[corte:].copy()
    
    modelo_v2 = treinar_modelo(treino_v2)
    teste_v2['predicao'] = teste_v2['tokens'].apply(lambda x: classificar_texto(x, modelo_v2))
    
    acc_v2 = accuracy_score(teste_v2['sentimento'], teste_v2['predicao'])
    diff_v2 = acc_v2 - acc_baseline
    resultados['V2'] = classification_report(teste_v2['sentimento'], teste_v2['predicao'], output_dict=True)
    
    print(f"📊 Acurácia V2: {acc_v2:.2%} (Diferença: {diff_v2:+.2%})")
    salvar_matriz(teste_v2['sentimento'], teste_v2['predicao'], 
                  f"Matriz V2 (Negação) - Acc: {acc_v2:.2%}", 
                  "matriz_v2.png", cmap='Greens')

    # --- V3: STEMMING ---
    print("\n" + "="*50)
    print("🚀 Executando V3 (Stemming)...")
    df_v3 = df_embaralhado.copy()
    df_v3['tokens'] = df_v3['texto_completo'].apply(limpeza_v3_stemming)
    
    treino_v3 = df_v3.iloc[:corte]
    teste_v3 = df_v3.iloc[corte:].copy()
    
    modelo_v3 = treinar_modelo(treino_v3)
    teste_v3['predicao'] = teste_v3['tokens'].apply(lambda x: classificar_texto(x, modelo_v3))
    
    acc_v3 = accuracy_score(teste_v3['sentimento'], teste_v3['predicao'])
    diff_v3 = acc_v3 - acc_baseline
    resultados['V3'] = classification_report(teste_v3['sentimento'], teste_v3['predicao'], output_dict=True)
    
    print(f"📊 Acurácia V3: {acc_v3:.2%} (Diferença: {diff_v3:+.2%})")
    salvar_matriz(teste_v3['sentimento'], teste_v3['predicao'], 
                  f"Matriz V3 (Stemming) - Acc: {acc_v3:.2%}", 
                  "matriz_v3.png", cmap='Oranges')

    # --- V4: BALANCEAMENTO ---
    print("\n" + "="*50)
    print("🚀 Executando V4 (Balanceado)...")
    df_v4 = df_v2.copy() 
    treino_v4_raw = df_v4.iloc[:corte]
    treino_v4_bal = balancear_dados_treino(treino_v4_raw)
    teste_v4 = df_v4.iloc[corte:].copy()
    
    modelo_v4 = treinar_modelo(treino_v4_bal)
    teste_v4['predicao'] = teste_v4['tokens'].apply(lambda x: classificar_texto(x, modelo_v4))
    
    acc_v4 = accuracy_score(teste_v4['sentimento'], teste_v4['predicao'])
    diff_v4 = acc_v4 - acc_baseline
    resultados['V4'] = classification_report(teste_v4['sentimento'], teste_v4['predicao'], output_dict=True)
    
    print(f"📊 Acurácia V4: {acc_v4:.2%} (Diferença: {diff_v4:+.2%})")
    salvar_matriz(teste_v4['sentimento'], teste_v4['predicao'], 
                  f"Matriz V4 (Balanceada) - Acc: {acc_v4:.2%}", 
                  "matriz_v4.png", cmap='Purples')

    # --- VISUALIZAÇÃO FINAL (RECALL) ---
    print("\n📊 Gerando Gráfico Comparativo Final...")
    
    classes = ['Positivo', 'Negativo', 'Neutro']
    dados_comp = []
    for versao, report in resultados.items():
        for cls in classes:
            dados_comp.append({
                'Versão': versao,
                'Classe': cls,
                'Recall': report[cls]['recall']
            })
            
    df_plot = pd.DataFrame(dados_comp)
    plt.figure(figsize=(12, 6))
    grafico = sns.barplot(data=df_plot, x='Classe', y='Recall', hue='Versão', palette='viridis')
    plt.title('Comparativo Final de Recall (Capacidade de Detecção)')
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.3)
    plt.ylim(0, 1.15)
    for container in grafico.containers:
        grafico.bar_label(container, fmt='%.2f')
    
    # Salvar o gráfico final
    caminho_final = os.path.join(OUTPUT_DIR, "comparativo_recall.png")
    plt.savefig(caminho_final)
    print(f"💾 Gráfico final salvo em: {caminho_final}")
    print("\n✅ Processo concluído! Verifique a pasta 'assets'.")

if __name__ == "__main__":
    executar_analise()