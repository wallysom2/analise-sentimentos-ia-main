"""Classe base para todos os classificadores de sentimento."""
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


class BaseClassifier(ABC):
    """Interface base para classificadores de sentimento."""
    
    @abstractmethod
    def train(self, df) -> None:
        """Treina o modelo com os dados fornecidos."""
        pass
    
    @abstractmethod
    def predict(self, text: str) -> str:
        """Retorna a classificação de sentimento."""
        pass
    
    @abstractmethod
    def predict_with_confidence(self, text: str) -> Tuple[str, float]:
        """Retorna a classificação e a confiança."""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo."""
        pass
