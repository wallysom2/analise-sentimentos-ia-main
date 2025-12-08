# Modelos Treinados

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
