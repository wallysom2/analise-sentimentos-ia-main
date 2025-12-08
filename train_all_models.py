"""
Script para treinar todos os modelos de uma vez e salvá-los
Executar este script UMA vez, depois só usar os modelos salvos
"""
import pandas as pd
import pickle
import os
from datetime import datetime
from src.classifiers import NaiveBayesClassifier, SGDSentimentClassifier, BayesianNetworkClassifier

# ==================== CONFIGURAÇÃO ====================

DATA_PATH = 'data/base-reviews-b2w.csv'
MODELS_DIR = 'classifiers'

# Cria pasta de modelos se não existir
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
    print(f"📂 Pasta '{MODELS_DIR}' criada!")

# ==================== FUNÇÕES ====================

def carregar_dados():
    """Carrega e prepara o dataset."""
    print("\n📂 Carregando dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df[['overall_rating', 'review_title', 'review_text']].copy()
    df['review_title'] = df['review_title'].fillna('')
    df = df.dropna(subset=['review_text'])
    df['texto_completo'] = df['review_title'] + " " + df['review_text']
    
    def definir_sentimento(nota):
        if nota >= 4:
            return 'Positivo'
        elif nota == 3:
            return 'Neutro'
        else:
            return 'Negativo'
    
    df['sentimento'] = df['overall_rating'].apply(definir_sentimento)
    print(f"✅ Dataset carregado: {len(df)} registros")
    print(f"   - Positivos: {len(df[df['sentimento'] == 'Positivo'])}")
    print(f"   - Negativos: {len(df[df['sentimento'] == 'Negativo'])}")
    print(f"   - Neutros: {len(df[df['sentimento'] == 'Neutro'])}")
    return df

def salvar_modelo(modelo, nome, info_adicional=None):
    """Salva modelo treinado em arquivo pickle."""
    filepath = os.path.join(MODELS_DIR, f"{nome}.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(modelo, f)
    
    # Salva informações do modelo
    info_filepath = os.path.join(MODELS_DIR, f"{nome}_info.txt")
    with open(info_filepath, 'w', encoding='utf-8') as f:
        f.write(f"Modelo: {nome}\n")
        f.write(f"Data de treinamento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Treinado: {modelo.is_trained}\n")
        if info_adicional:
            f.write(f"\nInformações adicionais:\n")
            for key, value in info_adicional.items():
                f.write(f"  {key}: {value}\n")
    
    print(f"   ✅ Salvo: {filepath}")

def treinar_naive_bayes(df):
    """Treina e salva Naive Bayes."""
    print("\n🧠 Treinando Naive Bayes...")
    print("   Configuração: preprocessamento com negação")
    
    modelo = NaiveBayesClassifier(preprocessing="negation")
    
    # Pré-processa os tokens
    df_train = df.copy()
    df_train['tokens'] = df_train['texto_completo'].apply(modelo._tokenize)
    
    # Treina
    modelo.train(df_train)
    
    # Salva
    info = modelo.get_info()
    salvar_modelo(modelo, "naive_bayes_model", info)
    
    return modelo

def treinar_sgd(df):
    """Treina e salva SGD Classifier."""
    print("\n🧠 Treinando SGD Classifier...")
    print("   Configuração: TF-IDF com reforço de classe neutra")
    
    modelo = SGDSentimentClassifier(neutral_weight_boost=2.5)
    modelo.train(df)
    
    # Salva
    info = modelo.get_info()
    salvar_modelo(modelo, "sgd_model", info)
    
    return modelo

def treinar_bayesian_network(df):
    """Treina e salva Rede Bayesiana."""
    print("\n🧠 Treinando Rede Bayesiana...")
    print("   ⚠️ AVISO: Este modelo é mais lento para treinar")
    print("   Configuração: 30 palavras-chave, 10% dos dados")
    
    try:
        modelo = BayesianNetworkClassifier(n_top_words=30, sample_fraction=0.1)
        modelo.train(df)
        
        # Salva
        info = modelo.get_info()
        salvar_modelo(modelo, "bayesian_network_model", info)
        
        return modelo
    except ImportError:
        print("   ❌ ERRO: pgmpy não instalado")
        print("   Execute: pip install pgmpy")
        return None

def testar_modelos(modelos, df):
    """Testa todos os modelos com alguns exemplos."""
    print("\n\n🧪 TESTANDO MODELOS")
    print("=" * 60)
    
    textos_teste = [
        "Este produto é excelente! Recomendo muito!",
        "Péssimo atendimento, nunca mais compro aqui.",
        "O produto chegou no prazo, nada de especial.",
        "Adorei! Melhor compra que já fiz!",
        "Não gostei, qualidade ruim."
    ]
    
    for i, texto in enumerate(textos_teste, 1):
        print(f"\n{'─' * 60}")
        print(f"Teste {i}: \"{texto[:50]}...\"" if len(texto) > 50 else f"Teste {i}: \"{texto}\"")
        print(f"{'─' * 60}")
        
        for nome, modelo in modelos.items():
            if modelo is None:
                continue
            
            try:
                resultado = modelo.predict_with_details(texto)
                sentimento = resultado['sentiment']
                confianca = resultado['confidence']
                print(f"  {nome:20} → {sentimento:10} (confiança: {confianca:.2%})")
            except Exception as e:
                print(f"  {nome:20} → ERRO: {str(e)[:40]}")

def criar_readme():
    """Cria README na pasta de modelos."""
    readme_path = os.path.join(MODELS_DIR, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("""# Modelos Treinados

Esta pasta contém os modelos de Machine Learning treinados para análise de sentimentos.

## Modelos Disponíveis

### 1. Naive Bayes (`naive_bayes_model.pkl`)
- **Descrição**: Implementação do zero com suavização de Laplace
- **Preprocessamento**: Tratamento de negações
- **Uso**: Modelo padrão, rápido e eficiente
- **Tamanho**: ~100KB

### 2. SGD Classifier (`sgd_model.pkl`)
- **Descrição**: Gradient Descent Otimizado com TF-IDF
- **Preprocessamento**: TF-IDF com n-gramas (1-3)
- **Uso**: Alta performance, bom para grandes volumes
- **Tamanho**: ~5MB

### 3. Bayesian Network (`bayesian_network_model.pkl`)
- **Descrição**: Rede Bayesiana com pgmpy
- **Preprocessamento**: TF-IDF + seleção de features
- **Uso**: Modelo probabilístico avançado
- **Tamanho**: ~500KB
- **Requisito**: pgmpy

## Como Usar

### Carregar Modelo
```python
import pickle

# Carregar modelo
with open('classifiers/naive_bayes_model.pkl', 'rb') as f:
    modelo = pickle.load(f)

# Fazer predição
resultado = modelo.predict_with_details("Texto para analisar")
print(resultado['sentiment'])  # Positivo/Negativo/Neutro
print(resultado['confidence'])  # 0.85 (85%)
```

### Re-treinar Modelos
Se precisar re-treinar (novos dados, ajustes):
```bash
python train_all_models.py
```

## Informações de Treinamento

Cada modelo tem um arquivo `*_info.txt` com:
- Data de treinamento
- Configurações usadas
- Estatísticas do modelo

## Observações

- ✅ Modelos são carregados automaticamente pela API
- ✅ Não precisa re-treinar a cada uso
- ✅ Re-treine apenas quando tiver novos dados ou quiser ajustar
- ⚠️ Arquivos .pkl devem ser mantidos nesta pasta
""")
    print(f"\n📄 README criado: {readme_path}")

# ==================== EXECUÇÃO PRINCIPAL ====================

def main():
    print("=" * 60)
    print("🚀 TREINAMENTO DE TODOS OS MODELOS")
    print("=" * 60)
    print("\nEste script vai:")
    print("  1. Carregar o dataset")
    print("  2. Treinar 3 modelos diferentes")
    print("  3. Salvar todos em 'classifiers/'")
    print("  4. Testar os modelos")
    print("\n⏱️ Tempo estimado: 2-5 minutos")
    print("=" * 60)
    
    input("\nPressione ENTER para começar ou CTRL+C para cancelar...")
    
    try:
        # 1. Carregar dados
        df = carregar_dados()
        
        # 2. Treinar modelos
        print("\n" + "=" * 60)
        print("FASE 1: TREINAMENTO DOS MODELOS")
        print("=" * 60)
        
        modelos = {}
        
        modelos['Naive Bayes'] = treinar_naive_bayes(df)
        modelos['SGD Classifier'] = treinar_sgd(df)
        modelos['Bayesian Network'] = treinar_bayesian_network(df)
        
        # 3. Criar README
        criar_readme()
        
        # 4. Testar modelos
        testar_modelos(modelos, df)
        
        # 5. Resumo final
        print("\n\n" + "=" * 60)
        print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("=" * 60)
        print("\n📦 Modelos salvos em 'classifiers/':")
        for filename in os.listdir(MODELS_DIR):
            if filename.endswith('.pkl'):
                filepath = os.path.join(MODELS_DIR, filename)
                size = os.path.getsize(filepath) / 1024  # KB
                print(f"   - {filename:30} ({size:.1f} KB)")
        
        print("\n🚀 Próximos passos:")
        print("   1. Inicie a API: python run_api.py")
        print("   2. Inicie o frontend: streamlit run frontend_streamlit.py")
        print("   3. Os modelos serão carregados automaticamente!")
        
        print("\n💡 Dica: Você só precisa rodar este script novamente se:")
        print("   - Tiver novos dados de treinamento")
        print("   - Quiser ajustar hiperparâmetros")
        print("   - Os modelos estiverem com performance ruim")
        
    except KeyboardInterrupt:
        print("\n\n❌ Treinamento cancelado pelo usuário.")
    except Exception as e:
        print(f"\n\n❌ ERRO durante o treinamento:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
