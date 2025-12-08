"""
Classificador SGD - Gradient Descent Otimizado
Alternativa de alta performance usando sklearn.
"""
import re
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

from .base import BaseClassifier

MAX_FEATURES = 15000
NGRAM_RANGE = (1, 3)

portuguese_stopwords = set([
    'a', 'o', 'e', 'é', 'de', 'do', 'da', 'em', 'um', 'uma', 'os', 'as', 'que', 'para',
    'com', 'se', 'por', 'no', 'na', 'ao', 'aos', 'meu', 'minha', 'nosso', 'nossa', 'ele', 
    'ela', 'você', 'dele', 'dela', 'isso', 'isto', 'aquele', 'aquela', 'mas', 'ou', 'já', 
    'também', 'ser', 'ter', 'foi', 'pelo', 'pela', 'só', 'mais', 'menos', 
    'muito', 'pouco', 'nada', 'sem'
])


class SGDSentimentClassifier(BaseClassifier):
    """
    Classificador baseado em SGDClassifier com TF-IDF.
    Otimizado para grandes volumes de dados.
    """
    
    def __init__(self, neutral_weight_boost: float = 2.5):
        """
        Args:
            neutral_weight_boost: Multiplicador do peso da classe Neutra
        """
        self.neutral_weight_boost = neutral_weight_boost
        self.pipeline = None
        self.is_trained = False
        self.n_features = 0
        
    def _preprocess_text(self, text: str) -> str:
        """Pré-processa o texto para o vetorizador."""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Záéíóúãõç\s]', ' ', text)
        tokens = text.split()
        return " ".join([w for w in tokens if w not in portuguese_stopwords and len(w) > 2])
    
    def _categorize_rating(self, rating) -> str:
        """Converte rating numérico para categoria."""
        if rating in [1, 2]:
            return "Negativo"
        if rating == 3:
            return "Neutro"
        if rating in [4, 5]:
            return "Positivo"
        return "Desconhecido"
    
    def train(self, df) -> None:
        """
        Treina o modelo SGD.
        
        Args:
            df: DataFrame com 'complete_review'/'texto_completo' e 
                'overall_rating'/'sentimento'
        """
        data = df.copy()
        
        # Determina a coluna de texto
        if 'complete_review' in data.columns:
            text_col = 'complete_review'
        elif 'texto_completo' in data.columns:
            text_col = 'texto_completo'
        else:
            raise ValueError("DataFrame deve ter 'complete_review' ou 'texto_completo'")
        
        # Determina a coluna de sentimento
        if 'sentimento' in data.columns:
            data['Sentiment_Target'] = data['sentimento']
        elif 'overall_rating' in data.columns:
            data['Sentiment_Target'] = data['overall_rating'].apply(self._categorize_rating)
        else:
            raise ValueError("DataFrame deve ter 'sentimento' ou 'overall_rating'")
        
        data = data[data['Sentiment_Target'] != 'Desconhecido'].copy()
        data['clean_review'] = data[text_col].apply(self._preprocess_text)
        
        X = data['clean_review']
        y = data['Sentiment_Target']
        
        # Cálculo de pesos balanceados
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        classes = le.classes_
        
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        
        # Reforça peso da classe Neutra
        neutral_idx = np.where(classes == 'Neutro')[0]
        if len(neutral_idx) > 0:
            weights[neutral_idx[0]] *= self.neutral_weight_boost
        
        class_weights_dict = dict(zip(classes, weights))
        
        sgd_clf = SGDClassifier(
            loss='log_loss',
            penalty='elasticnet',
            alpha=0.0001,
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            class_weight=class_weights_dict
        )
        
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=NGRAM_RANGE,
                max_features=MAX_FEATURES
            )),
            ('clf', sgd_clf)
        ])
        
        self.pipeline.fit(X, y)
        self.n_features = len(self.pipeline['tfidf'].get_feature_names_out())
        self.is_trained = True
    
    def predict(self, text: str) -> str:
        """Classifica um texto."""
        if not self.is_trained:
            raise RuntimeError("Modelo não treinado. Chame train() primeiro.")
        
        processed = self._preprocess_text(text)
        return self.pipeline.predict([processed])[0]
    
    def predict_with_confidence(self, text: str) -> Tuple[str, float]:
        """Classifica um texto e retorna a confiança."""
        if not self.is_trained:
            raise RuntimeError("Modelo não treinado. Chame train() primeiro.")
        
        processed = self._preprocess_text(text)
        prediction = self.pipeline.predict([processed])[0]
        proba = self.pipeline.predict_proba([processed])[0]
        confidence = float(np.max(proba))
        
        return prediction, confidence
    
    def predict_with_details(self, text: str) -> Dict[str, Any]:
        """Classificação completa com todos os detalhes."""
        if not self.is_trained:
            raise RuntimeError("Modelo não treinado. Chame train() primeiro.")
        
        processed = self._preprocess_text(text)
        prediction = self.pipeline.predict([processed])[0]
        proba = self.pipeline.predict_proba([processed])[0]
        classes = self.pipeline.classes_
        
        probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        
        return {
            'sentiment': prediction,
            'confidence': float(np.max(proba)),
            'probabilities': probabilities,
            'preprocessed_text': processed
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo."""
        return {
            'name': 'SGD Classifier',
            'type': 'sgd',
            'is_trained': self.is_trained,
            'n_features': self.n_features,
            'neutral_weight_boost': self.neutral_weight_boost,
            'description': 'SGDClassifier com TF-IDF e pesos balanceados'
        }
