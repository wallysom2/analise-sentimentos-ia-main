"""
Classificador Naive Bayes - Implementação Original
Modelo principal para análise de sentimentos.
"""
import re
import math
from collections import defaultdict
from typing import Tuple, Dict, Any, List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from .base import BaseClassifier

# Downloads necessários do NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('rslp', quiet=True)

stop_words = set(stopwords.words('portuguese'))
stemmer = nltk.stem.RSLPStemmer()


class NaiveBayesClassifier(BaseClassifier):
    """
    Classificador Naive Bayes implementado do zero.
    Modelo original do projeto para análise de sentimentos em português.
    """
    
    def __init__(self, preprocessing: str = "negation"):
        """
        Args:
            preprocessing: Tipo de pré-processamento 
                          - "simple": Limpeza básica
                          - "negation": Trata negações (padrão)
                          - "stemming": Aplica stemming + negações
        """
        self.preprocessing = preprocessing
        self.model = None
        self.is_trained = False
        
    def _clean_simple(self, text: str) -> List[str]:
        """Limpeza V1: Básica."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        return [t for t in tokens if t not in stop_words]
    
    def _clean_negation(self, text: str) -> List[str]:
        """Limpeza V2: Com tratamento de negação."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        new_tokens = []
        skip = False
        
        for i in range(len(tokens)):
            if skip:
                skip = False
                continue
            t = tokens[i]
            if t == 'não':
                if i + 1 < len(tokens):
                    new_tokens.append("não_" + tokens[i + 1])
                    skip = True
                else:
                    new_tokens.append(t)
            elif t not in stop_words:
                new_tokens.append(t)
        return new_tokens
    
    def _clean_stemming(self, text: str) -> List[str]:
        """Limpeza V3: Com stemming e negação."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        new_tokens = []
        skip = False
        
        for i in range(len(tokens)):
            if skip:
                skip = False
                continue
            t = tokens[i]
            if t == 'não':
                if i + 1 < len(tokens):
                    stem_next = stemmer.stem(tokens[i + 1])
                    new_tokens.append("não_" + stem_next)
                    skip = True
                else:
                    new_tokens.append(t)
            elif t not in stop_words:
                new_tokens.append(stemmer.stem(t))
        return new_tokens
    
    def _tokenize(self, text: str) -> List[str]:
        """Aplica o pré-processamento configurado."""
        if self.preprocessing == "simple":
            return self._clean_simple(text)
        elif self.preprocessing == "stemming":
            return self._clean_stemming(text)
        else:  # negation (default)
            return self._clean_negation(text)
    
    def train(self, df) -> None:
        """
        Treina o modelo Naive Bayes.
        
        Args:
            df: DataFrame com colunas 'tokens' e 'sentimento'
                ou 'texto_completo' e 'sentimento'
        """
        count_pos = defaultdict(int)
        count_neg = defaultdict(int)
        count_neu = defaultdict(int)
        total_pos = total_neg = total_neu = 0
        vocabulary = set()
        
        for _, row in df.iterrows():
            # Suporta tanto tokens pré-processados quanto texto bruto
            if 'tokens' in df.columns:
                tokens = row['tokens']
            else:
                tokens = self._tokenize(row['texto_completo'])
                
            sentiment = row['sentimento']
            
            for token in tokens:
                vocabulary.add(token)
                if sentiment == 'Positivo':
                    count_pos[token] += 1
                    total_pos += 1
                elif sentiment == 'Negativo':
                    count_neg[token] += 1
                    total_neg += 1
                elif sentiment == 'Neutro':
                    count_neu[token] += 1
                    total_neu += 1
        
        self.model = {
            'c_pos': count_pos, 'c_neg': count_neg, 'c_neu': count_neu,
            't_pos': total_pos, 't_neg': total_neg, 't_neu': total_neu,
            'V': len(vocabulary)
        }
        self.is_trained = True
    
    def _calculate_scores(self, tokens: List[str]) -> Dict[str, float]:
        """Calcula os scores log-probabilísticos para cada classe."""
        c_pos = self.model['c_pos']
        c_neg = self.model['c_neg']
        c_neu = self.model['c_neu']
        t_pos = self.model['t_pos']
        t_neg = self.model['t_neg']
        t_neu = self.model['t_neu']
        V = self.model['V']
        
        s_pos = s_neg = s_neu = 0
        
        for token in tokens:
            s_pos += math.log((c_pos.get(token, 0) + 1) / (t_pos + V))
            s_neg += math.log((c_neg.get(token, 0) + 1) / (t_neg + V))
            s_neu += math.log((c_neu.get(token, 0) + 1) / (t_neu + V))
        
        return {'Positivo': s_pos, 'Negativo': s_neg, 'Neutro': s_neu}
    
    def predict(self, text: str) -> str:
        """
        Classifica um texto.
        
        Args:
            text: Texto para classificar
            
        Returns:
            Sentimento predito: 'Positivo', 'Negativo' ou 'Neutro'
        """
        if not self.is_trained:
            raise RuntimeError("Modelo não treinado. Chame train() primeiro.")
        
        tokens = self._tokenize(text)
        scores = self._calculate_scores(tokens)
        return max(scores, key=scores.get)
    
    def predict_with_confidence(self, text: str) -> Tuple[str, float]:
        """
        Classifica um texto e retorna a confiança.
        
        Args:
            text: Texto para classificar
            
        Returns:
            Tupla (sentimento, confiança normalizada)
        """
        if not self.is_trained:
            raise RuntimeError("Modelo não treinado. Chame train() primeiro.")
        
        tokens = self._tokenize(text)
        scores = self._calculate_scores(tokens)
        
        # Converte log-probabilidades para probabilidades normalizadas
        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        probabilities = {k: v / total for k, v in exp_scores.items()}
        
        best_sentiment = max(probabilities, key=probabilities.get)
        confidence = probabilities[best_sentiment]
        
        return best_sentiment, confidence
    
    def predict_with_details(self, text: str) -> Dict[str, Any]:
        """
        Classificação completa com todos os detalhes.
        
        Args:
            text: Texto para classificar
            
        Returns:
            Dict com sentimento, confiança, tokens e probabilidades
        """
        if not self.is_trained:
            raise RuntimeError("Modelo não treinado. Chame train() primeiro.")
        
        tokens = self._tokenize(text)
        scores = self._calculate_scores(tokens)
        
        # Normaliza para probabilidades
        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        probabilities = {k: round(v / total, 4) for k, v in exp_scores.items()}
        
        best_sentiment = max(probabilities, key=probabilities.get)
        
        return {
            'sentiment': best_sentiment,
            'confidence': probabilities[best_sentiment],
            'probabilities': probabilities,
            'tokens': tokens,
            'preprocessing': self.preprocessing
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo."""
        return {
            'name': 'Naive Bayes',
            'type': 'naive_bayes',
            'preprocessing': self.preprocessing,
            'is_trained': self.is_trained,
            'vocabulary_size': self.model['V'] if self.is_trained else 0,
            'description': 'Classificador Naive Bayes implementado do zero com suavização de Laplace'
        }
