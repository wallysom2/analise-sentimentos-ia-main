"""
API REST para Análise de Sentimentos
Suporta múltiplos classificadores: Naive Bayes (padrão), SGD e Rede Bayesiana
"""
import os
import pandas as pd
from typing import Optional, List
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from classifiers import NaiveBayesClassifier, SGDSentimentClassifier, BayesianNetworkClassifier

# ==================== CONFIGURAÇÃO ====================

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'base-reviews-b2w.csv')

app = FastAPI(
    title="API de Análise de Sentimentos",
    description="""
    API para classificação de sentimentos em textos em português.
    
    ## Classificadores Disponíveis
    
    - **naive_bayes** (padrão): Implementação do zero com suavização de Laplace
    - **sgd**: SGDClassifier otimizado com TF-IDF
    - **bayesian_network**: Rede Bayesiana (requer pgmpy)
    
    ## Uso
    
    1. O modelo Naive Bayes é carregado automaticamente na inicialização
    2. Use `/classify` para classificar textos
    3. Use `/classify/batch` para classificar múltiplos textos
    """,
    version="1.0.0"
)

# CORS para permitir requisições de outras origens
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== MODELOS PYDANTIC ====================

class ClassifierType(str, Enum):
    naive_bayes = "naive_bayes"
    sgd = "sgd"
    bayesian_network = "bayesian_network"


class PreprocessingType(str, Enum):
    simple = "simple"
    negation = "negation"
    stemming = "stemming"


class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Texto para classificar", min_length=1)
    classifier: ClassifierType = Field(
        default=ClassifierType.naive_bayes,
        description="Tipo de classificador a usar"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "O produto é excelente, recomendo!",
                "classifier": "naive_bayes"
            }
        }


class BatchClassifyRequest(BaseModel):
    texts: List[str] = Field(..., description="Lista de textos para classificar", min_length=1)
    classifier: ClassifierType = Field(default=ClassifierType.naive_bayes)
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "Produto excelente!",
                    "Péssimo atendimento",
                    "Entrega no prazo"
                ],
                "classifier": "naive_bayes"
            }
        }


class ClassifyResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: dict
    classifier_used: str


class BatchClassifyResponse(BaseModel):
    results: List[ClassifyResponse]
    classifier_used: str
    total_processed: int


class ModelInfoResponse(BaseModel):
    available_classifiers: List[str]
    current_models: dict
    

# ==================== ESTADO GLOBAL ====================

class ModelManager:
    """Gerencia os modelos carregados."""
    
    def __init__(self):
        self.naive_bayes: Optional[NaiveBayesClassifier] = None
        self.sgd: Optional[SGDSentimentClassifier] = None
        self.bayesian_network: Optional[BayesianNetworkClassifier] = None
        self.df: Optional[pd.DataFrame] = None
    
    def load_data(self):
        """Carrega os dados do CSV."""
        if self.df is None:
            print("📂 Carregando dataset...")
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
            self.df = df
            print(f"✅ Dataset carregado: {len(df)} registros")
        return self.df
    
    def get_classifier(self, classifier_type: ClassifierType):
        """Retorna o classificador solicitado, treinando se necessário."""
        df = self.load_data()
        
        if classifier_type == ClassifierType.naive_bayes:
            if self.naive_bayes is None:
                print("🔄 Treinando Naive Bayes...")
                self.naive_bayes = NaiveBayesClassifier(preprocessing="negation")
                # Pré-processa os tokens
                df_train = df.copy()
                df_train['tokens'] = df_train['texto_completo'].apply(
                    self.naive_bayes._tokenize
                )
                self.naive_bayes.train(df_train)
                print("✅ Naive Bayes pronto!")
            return self.naive_bayes
        
        elif classifier_type == ClassifierType.sgd:
            if self.sgd is None:
                print("🔄 Treinando SGD Classifier...")
                self.sgd = SGDSentimentClassifier()
                self.sgd.train(df)
                print("✅ SGD Classifier pronto!")
            return self.sgd
        
        elif classifier_type == ClassifierType.bayesian_network:
            if self.bayesian_network is None:
                print("🔄 Treinando Rede Bayesiana (isso pode demorar)...")
                self.bayesian_network = BayesianNetworkClassifier()
                self.bayesian_network.train(df)
                print("✅ Rede Bayesiana pronta!")
            return self.bayesian_network


model_manager = ModelManager()


# ==================== EVENTOS ====================

@app.on_event("startup")
async def startup_event():
    """Carrega o modelo Naive Bayes na inicialização."""
    print("\n🚀 Iniciando API de Análise de Sentimentos...")
    try:
        model_manager.get_classifier(ClassifierType.naive_bayes)
        print("✅ API pronta para receber requisições!\n")
    except Exception as e:
        print(f"⚠️ Erro ao carregar modelo: {e}")
        print("A API iniciará, mas você precisará verificar o caminho do dataset.")


# ==================== ENDPOINTS ====================

@app.get("/", tags=["Info"])
async def root():
    """Endpoint raiz com informações da API."""
    return {
        "name": "API de Análise de Sentimentos",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs"
    }


@app.get("/health", tags=["Info"])
async def health_check():
    """Verifica o status da API."""
    return {
        "status": "healthy",
        "models_loaded": {
            "naive_bayes": model_manager.naive_bayes is not None,
            "sgd": model_manager.sgd is not None,
            "bayesian_network": model_manager.bayesian_network is not None
        }
    }


@app.get("/models", response_model=ModelInfoResponse, tags=["Info"])
async def get_models_info():
    """Retorna informações sobre os modelos disponíveis."""
    models_info = {}
    
    if model_manager.naive_bayes:
        models_info["naive_bayes"] = model_manager.naive_bayes.get_info()
    if model_manager.sgd:
        models_info["sgd"] = model_manager.sgd.get_info()
    if model_manager.bayesian_network:
        models_info["bayesian_network"] = model_manager.bayesian_network.get_info()
    
    return {
        "available_classifiers": ["naive_bayes", "sgd", "bayesian_network"],
        "current_models": models_info
    }


@app.post("/classify", response_model=ClassifyResponse, tags=["Classificação"])
async def classify_text(request: ClassifyRequest):
    """
    Classifica um texto e retorna o sentimento.
    
    - **text**: Texto para análise
    - **classifier**: Classificador a usar (naive_bayes, sgd, bayesian_network)
    """
    try:
        classifier = model_manager.get_classifier(request.classifier)
        result = classifier.predict_with_details(request.text)
        
        return ClassifyResponse(
            text=request.text,
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            classifier_used=request.classifier.value
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/batch", response_model=BatchClassifyResponse, tags=["Classificação"])
async def classify_batch(request: BatchClassifyRequest):
    """
    Classifica múltiplos textos de uma vez.
    
    - **texts**: Lista de textos para análise
    - **classifier**: Classificador a usar
    """
    try:
        classifier = model_manager.get_classifier(request.classifier)
        results = []
        
        for text in request.texts:
            result = classifier.predict_with_details(text)
            results.append(ClassifyResponse(
                text=text,
                sentiment=result['sentiment'],
                confidence=result['confidence'],
                probabilities=result['probabilities'],
                classifier_used=request.classifier.value
            ))
        
        return BatchClassifyResponse(
            results=results,
            classifier_used=request.classifier.value,
            total_processed=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train/{classifier_type}", tags=["Treinamento"])
async def train_model(classifier_type: ClassifierType):
    """
    Força o (re)treinamento de um modelo específico.
    
    Útil para recarregar modelos após atualização dos dados.
    """
    try:
        # Reseta o modelo para forçar retreino
        if classifier_type == ClassifierType.naive_bayes:
            model_manager.naive_bayes = None
        elif classifier_type == ClassifierType.sgd:
            model_manager.sgd = None
        elif classifier_type == ClassifierType.bayesian_network:
            model_manager.bayesian_network = None
        
        # Retreina
        classifier = model_manager.get_classifier(classifier_type)
        info = classifier.get_info()
        
        return {
            "message": f"Modelo {classifier_type.value} treinado com sucesso",
            "model_info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== EXECUÇÃO ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
