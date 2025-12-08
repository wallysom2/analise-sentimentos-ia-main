# 🚀 Guia: Treinar Uma Vez, Usar Sempre

## 📋 Resumo

Agora você pode **treinar todos os modelos uma única vez** e depois usá-los quantas vezes quiser, sem precisar re-treinar!

## 🎯 Fluxo de Trabalho

### **1️⃣ Treinar Todos os Modelos (FAÇA UMA VEZ)**

```bash
python train_all_models.py
```

**O que isso faz:**
- ✅ Treina 3 modelos: Naive Bayes, SGD e Bayesian Network
- ✅ Salva cada um em `classifiers/*.pkl`
- ✅ Testa todos com exemplos
- ✅ Cria documentação automática
- ⏱️ Tempo: 2-5 minutos

**Modelos salvos:**
```
classifiers/
├── naive_bayes_model.pkl        (~100 KB)
├── sgd_model.pkl                (~5 MB)
├── bayesian_network_model.pkl   (~500 KB)
├── *_info.txt                   (informações de cada modelo)
└── README.md                    (documentação)
```

---

### **2️⃣ Iniciar a API (USA MODELOS SALVOS)**

```bash
python run_api.py
```

**O que acontece:**
- ⚡ Carrega modelos salvos (RÁPIDO!)
- ❌ NÃO treina novamente
- ✅ API pronta em segundos

---

### **3️⃣ Iniciar o Frontend**

```bash
streamlit run frontend_streamlit.py
```

**O que acontece:**
- 🎨 Interface abre no navegador
- 🔄 Conecta com a API
- 🚀 Pronto para usar!

---

## 🔄 Quando Treinar Novamente?

Só execute `python train_all_models.py` novamente quando:

- ✅ **Novos dados**: Você tem mais dados de treinamento
- ✅ **Ajustes**: Quer mudar hiperparâmetros
- ✅ **Performance**: Modelos com resultados ruins
- ✅ **Novo algoritmo**: Adicionar outro classificador

---

## 💡 Vantagens

| Antes | Depois |
|-------|--------|
| ❌ Treina a cada inicialização | ✅ Carrega em segundos |
| ❌ 2-5 minutos de espera | ✅ API pronta instantaneamente |
| ❌ Treina mesmo sem mudanças | ✅ Treina só quando necessário |
| ❌ Difícil trocar entre modelos | ✅ Troca instantânea |

---

## 🎮 Como Usar Diferentes Modelos

### **No Frontend Streamlit:**
Escolha no dropdown:
- Naive Bayes
- SGD
- Bayesian Network

### **Na API (via código):**
```python
import requests

# Naive Bayes
response = requests.post("http://localhost:8000/classify", 
    json={"text": "Produto ótimo!", "classifier": "naive_bayes"})

# SGD
response = requests.post("http://localhost:8000/classify", 
    json={"text": "Produto ótimo!", "classifier": "sgd"})

# Bayesian Network
response = requests.post("http://localhost:8000/classify", 
    json={"text": "Produto ótimo!", "classifier": "bayesian_network"})
```

---

## 🔧 Comandos Úteis

### Verificar status dos modelos
```bash
curl http://localhost:8000/health
```

### Re-treinar modelo específico (via API)
```bash
curl -X POST http://localhost:8000/train/sgd
```

### Listar modelos disponíveis
```bash
curl http://localhost:8000/models
```

---

## 📁 Estrutura Final

```
analise-sentimentos-ia-main/
│
├── train_all_models.py          ← 🆕 Treina todos os modelos
├── run_api.py                   ← Inicia API
├── frontend_streamlit.py        ← Frontend
│
├── classifiers/                 ← 🆕 Modelos salvos
│   ├── naive_bayes_model.pkl
│   ├── sgd_model.pkl
│   ├── bayesian_network_model.pkl
│   ├── *_info.txt
│   └── README.md
│
├── src/
│   ├── api.py                   ← API original (treina toda vez)
│   ├── api_with_saved_models.py ← 🆕 API otimizada (usa modelos salvos)
│   └── classifiers/
│       ├── naive_bayes.py
│       ├── sgd_classifier.py
│       └── bayesian_network.py
│
└── data/
    └── base-reviews-b2w.csv
```

---

## 🚀 Início Rápido (Setup Completo)

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Treinar todos os modelos (UMA VEZ)
python train_all_models.py

# 3. Iniciar API (em um terminal)
python run_api.py

# 4. Iniciar Frontend (em outro terminal)
streamlit run frontend_streamlit.py
```

**Pronto! 🎉**
- API: http://localhost:8000
- Frontend: http://localhost:8501
- Docs: http://localhost:8000/docs

---

## ❓ FAQ

### **Os modelos .pkl devem ir no Git?**
- ✅ **SIM**, se o repositório é privado e você quer compartilhar com a equipe
- ❌ **NÃO**, se são muito grandes ou você quer que cada um treine localmente
- 💡 **Dica**: Adicione `classifiers/*.pkl` no `.gitignore` se não quiser versionar

### **Quanto espaço ocupam?**
- Naive Bayes: ~100 KB
- SGD: ~5 MB
- Bayesian Network: ~500 KB
- **Total**: ~6 MB

### **Posso adicionar mais modelos?**
Sim! Edite `train_all_models.py` e adicione:
```python
from sklearn.svm import SVC

def treinar_svm(df):
    modelo = SVC(probability=True)
    # ... treinar ...
    salvar_modelo(modelo, "svm_model")
```

---

## 🎯 Resumo

1. **Treine UMA vez**: `python train_all_models.py`
2. **Use SEMPRE**: Modelos carregam automaticamente
3. **Troque FÁCIL**: Escolha entre 3 algoritmos
4. **Performance**: API inicia em segundos

**Nunca mais perca tempo treinando modelos que já estão prontos!** 🚀
