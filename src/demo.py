import pandas as pd
from utils import limpeza_v2_negacao, treinar_modelo, classificar_texto



print("\n🤖 INICIALIZANDO O AGENTE DE IA (V2)...")

try:
    print("⏳ 1/3 Lendo arquivo CSV...")
    df = pd.read_csv('data/base-reviews-b2w.csv')
    

    df['review_title'] = df['review_title'].fillna('')
    df = df.dropna(subset=['review_text'])
    df['texto_completo'] = df['review_title'] + " " + df['review_text']
    
    def definir_sentimento(nota):
        if nota >= 4: return 'Positivo'
        elif nota == 3: return 'Neutro'
        else: return 'Negativo'
    df['sentimento'] = df['overall_rating'].apply(definir_sentimento)

    print("⏳ 2/3 Processando Negações e Tokens...")

    df['tokens'] = df['texto_completo'].apply(limpeza_v2_negacao)

    print("⏳ 3/3 Treinando Modelo Probabilístico...")
    modelo = treinar_modelo(df) 
    
    print("✅ SISTEMA PRONTO! Modelo carregado na memória.\n")

except Exception as e:
    print(f"\n❌ Erro crítico ao carregar: {e}")
    print("Dica: Verifique se o arquivo 'base-reviews-b2w.csv' está na pasta 'data/'")
    exit()


print("="*60)
print("💬 CHAT DE ANÁLISE DE SENTIMENTOS")
print("Digite uma avaliação para a IA classificar.")
print("Comandos: digite 'sair' para fechar.")
print("="*60)

while True:
    frase = input("\n📝 Digite a frase: ")
    if frase.lower() in ['sair', 'exit']: 
        print("Encerrando...")
        break
    

    tokens = limpeza_v2_negacao(frase)
    predicao = classificar_texto(tokens, modelo)
    
    cor = ""
    icone = ""
    if predicao == 'Positivo': 
        cor = "\033[92m" 

    elif predicao == 'Negativo': 
        cor = "\033[91m" 

    elif predicao == 'Neutro': 
        cor = "\033[93m"   

    
    print(f"   Resultado: {cor} {predicao}\033[0m")
    print(f"   Tokens: {tokens}")