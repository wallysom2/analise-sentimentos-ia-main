import re
import math
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('rslp', quiet=True)

stop_words = set(stopwords.words('portuguese'))
stemmer = nltk.stem.RSLPStemmer()

# --- FUNÇÕES DE LIMPEZA (V1, V2, V3) ---

def limpeza_v1_simples(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    tokens = word_tokenize(texto)
    return [t for t in tokens if t not in stop_words]

def limpeza_v2_negacao(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    tokens = word_tokenize(texto)
    novos_tokens = []
    pular = False
    
    for i in range(len(tokens)):
        if pular:
            pular = False
            continue
        t = tokens[i]
        if t == 'não':
            if i+1 < len(tokens):
                novos_tokens.append("não_" + tokens[i+1])
                pular = True
            else:
                novos_tokens.append(t)
        elif t not in stop_words:
            novos_tokens.append(t)
    return novos_tokens

def limpeza_v3_stemming(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    tokens = word_tokenize(texto)
    novos_tokens = []
    pular = False
    
    for i in range(len(tokens)):
        if pular:
            pular = False
            continue
        t = tokens[i]
        if t == 'não':
            if i+1 < len(tokens):
                raiz_prox = stemmer.stem(tokens[i+1])
                novos_tokens.append("não_" + raiz_prox)
                pular = True
            else:
                novos_tokens.append(t)
        elif t not in stop_words:
            novos_tokens.append(stemmer.stem(t))
    return novos_tokens

# --- FUNÇÕES DO MODELO NAIVE BAYES ---

def treinar_modelo(df_treino):
    contagem_pos = defaultdict(int); contagem_neg = defaultdict(int); contagem_neu = defaultdict(int)
    total_pos = 0; total_neg = 0; total_neu = 0; vocabulario = set()

    for _, linha in df_treino.iterrows():
        tokens = linha['tokens']
        cat = linha['sentimento']
        for p in tokens:
            vocabulario.add(p)
            if cat == 'Positivo': contagem_pos[p]+=1; total_pos+=1
            elif cat == 'Negativo': contagem_neg[p]+=1; total_neg+=1
            elif cat == 'Neutro': contagem_neu[p]+=1; total_neu+=1

    return {
        'c_pos': contagem_pos, 'c_neg': contagem_neg, 'c_neu': contagem_neu,
        't_pos': total_pos, 't_neg': total_neg, 't_neu': total_neu,
        'V': len(vocabulario)
    }

def classificar_texto(tokens, modelo):
    c_pos = modelo['c_pos']; c_neg = modelo['c_neg']; c_neu = modelo['c_neu']
    t_pos = modelo['t_pos']; t_neg = modelo['t_neg']; t_neu = modelo['t_neu']
    V = modelo['V']

    s_pos = 0; s_neg = 0; s_neu = 0

    for p in tokens:
        s_pos += math.log((c_pos.get(p,0)+1)/(t_pos+V))
        s_neg += math.log((c_neg.get(p,0)+1)/(t_neg+V))
        s_neu += math.log((c_neu.get(p,0)+1)/(t_neu+V))

    m = max(s_pos, s_neg, s_neu)
    if m == s_pos: return 'Positivo'
    elif m == s_neg: return 'Negativo'
    else: return 'Neutro'