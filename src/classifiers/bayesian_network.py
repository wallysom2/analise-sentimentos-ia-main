"""
Classificador com Rede Bayesiana
Usa pgmpy para construir uma rede probabilística.
"""
import re
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import BaseClassifier

N_TOP_WORDS = 30

portuguese_stopwords = set([
    'a', 'o', 'e', 'é', 'de', 'do', 'da', 'em', 'um', 'uma', 'os', 'as', 'que', 'para',
    'com', 'se', 'por', 'no', 'na', 'ao', 'aos', 'meu', 'minha', 'nosso', 'nossa', 'ele', 
    'ela', 'você', 'dele', 'dela', 'isso', 'isto', 'aquele', 'aquela', 'mas', 'ou', 'já', 
    'também', 'ser', 'ter', 'foi', 'pelo', 'pela', 'só', 'mais', 'menos'
])


class BayesianNetworkClassifier(BaseClassifier):
    """
    Classificador baseado em Rede Bayesiana com pgmpy.
    Usa TF-IDF para selecionar features importantes.
    """
    
    def __init__(self, n_top_words: int = 30, sample_fraction: float = 0.1):
        """
        Args:
            n_top_words: Número de palavras mais importantes para usar
            sample_fraction: Fração dos dados para treino (redes bayesianas são lentas)
        """
        self.n_top_words = n_top_words
        self.sample_fraction = sample_fraction
        self.model = None
        self.top_words = []
        self.is_trained = False
        
    def _preprocess_text(self, text: str) -> str:
        """Pré-processa texto para o vetorizador."""
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
        Treina a rede bayesiana.
        
        NOTA: Requer pgmpy instalado. Este modelo é mais lento.
        """
        try:
            from pgmpy.estimators import HillClimbSearch, BayesianEstimator
            from pgmpy.models import DiscreteBayesianNetwork
        except ImportError:
            raise ImportError(
                "pgmpy não instalado. Execute: pip install pgmpy"
            )
        
        import pandas as pd
        data = df.copy()
        
        # Determina colunas
        if 'complete_review' in data.columns:
            text_col = 'complete_review'
        elif 'texto_completo' in data.columns:
            text_col = 'texto_completo'
        else:
            raise ValueError("DataFrame deve ter 'complete_review' ou 'texto_completo'")
        
        if 'sentimento' in data.columns:
            data['Sentiment_Target'] = data['sentimento'].astype('category')
        elif 'overall_rating' in data.columns:
            data['Sentiment_Target'] = data['overall_rating'].apply(
                self._categorize_rating
            ).astype('category')
        else:
            raise ValueError("DataFrame deve ter 'sentimento' ou 'overall_rating'")
        
        data = data[data['Sentiment_Target'] != 'Desconhecido'].copy()
        data['clean_review'] = data[text_col].apply(self._preprocess_text)
        
        # Extrai top palavras via TF-IDF
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(data['clean_review'])
        
        feature_names = vectorizer.get_feature_names_out()
        sum_tfidf = tfidf_matrix.sum(axis=0)
        tfidf_scores = [(feature_names[i], sum_tfidf[0, i]) 
                        for i in range(tfidf_matrix.shape[1])]
        
        sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        self.top_words = [word for word, score in sorted_scores[:self.n_top_words]]
        
        # Cria features binárias
        for word in self.top_words:
            col_name = f"Palavra_{word}_presente"
            data[col_name] = data['clean_review'].apply(
                lambda text: "Sim" if word in text else "Não"
            ).astype('category')
        
        # Prepara dados para treino
        feature_cols = [f"Palavra_{w}_presente" for w in self.top_words]
        train_cols = feature_cols + ['Sentiment_Target']
        
        # Amostra para performance
        sample_data = data[train_cols].sample(
            frac=self.sample_fraction, 
            random_state=42
        ).copy()
        
        # Treina a rede
        hc = HillClimbSearch(sample_data)
        best_model = hc.estimate(
            scoring_method="bdeu",
            max_indegree=4,
            epsilon=1e-4
        )
        
        self.model = DiscreteBayesianNetwork(best_model.edges())
        self.model.fit(sample_data, estimator=BayesianEstimator)
        self.is_trained = True
    
    def predict(self, text: str) -> str:
        """Classifica um texto."""
        prediction, _ = self.predict_with_confidence(text)
        return prediction
    
    def predict_with_confidence(self, text: str) -> Tuple[str, float]:
        """Classifica um texto e retorna a confiança."""
        if not self.is_trained:
            raise RuntimeError("Modelo não treinado. Chame train() primeiro.")
        
        try:
            from pgmpy.inference import VariableElimination
        except ImportError:
            raise ImportError("pgmpy não instalado.")
        
        clean_review = self._preprocess_text(text)
        evidence = {}
        
        for word in self.top_words:
            col_name = f"Palavra_{word}_presente"
            evidence[col_name] = "Sim" if word in clean_review else "Não"
        
        if not evidence:
            return "Indefinido", 0.0
        
        inference = VariableElimination(self.model)
        result = inference.query(
            variables=["Sentiment_Target"],
            evidence=evidence,
            show_progress=False
        )
        
        highest_prob = float(result.values.max())
        predicted_sentiment = result.state_names['Sentiment_Target'][
            np.argmax(result.values)
        ]
        
        return predicted_sentiment, highest_prob
    
    def predict_with_details(self, text: str) -> Dict[str, Any]:
        """Classificação completa com todos os detalhes."""
        if not self.is_trained:
            raise RuntimeError("Modelo não treinado. Chame train() primeiro.")
        
        try:
            from pgmpy.inference import VariableElimination
        except ImportError:
            raise ImportError("pgmpy não instalado.")
        
        clean_review = self._preprocess_text(text)
        evidence = {}
        words_found = []
        
        for word in self.top_words:
            col_name = f"Palavra_{word}_presente"
            is_present = word in clean_review
            evidence[col_name] = "Sim" if is_present else "Não"
            if is_present:
                words_found.append(word)
        
        if not evidence:
            return {
                'sentiment': "Indefinido",
                'confidence': 0.0,
                'probabilities': {},
                'words_found': []
            }
        
        inference = VariableElimination(self.model)
        result = inference.query(
            variables=["Sentiment_Target"],
            evidence=evidence,
            show_progress=False
        )
        
        probabilities = {
            state: round(float(prob), 4)
            for state, prob in zip(
                result.state_names['Sentiment_Target'],
                result.values
            )
        }
        
        predicted_sentiment = max(probabilities, key=probabilities.get)
        
        return {
            'sentiment': predicted_sentiment,
            'confidence': probabilities[predicted_sentiment],
            'probabilities': probabilities,
            'words_found': words_found,
            'top_words_used': self.top_words
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo."""
        return {
            'name': 'Bayesian Network',
            'type': 'bayesian_network',
            'is_trained': self.is_trained,
            'n_top_words': self.n_top_words,
            'sample_fraction': self.sample_fraction,
            'top_words': self.top_words if self.is_trained else [],
            'description': 'Rede Bayesiana com estrutura aprendida via Hill Climbing'
        }
