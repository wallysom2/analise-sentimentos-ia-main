"""
API REST para Análise de Sentimentos - VERSÃO COM MODELOS SALVOS
Carrega modelos pré-treinados ao invés de treinar a cada inicialização
"""
import os
import pickle
import pandas as pd
from typing import Optional, List
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from classifiers import NaiveBayesClassifier, SGDSentimentClassifier, BayesianNetworkClassifier

# ==================== CONFIGURAÇÃO ====================

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'base-reviews-b2w.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'classifiers')

app = FastAPI(
    title="API de Análise de Sentimentos",
    description="""
    API para classificação de sentimentos em textos em português.
    
    ## Classificadores Disponíveis
    
    - **naive_bayes**: Implementação do zero com suavização de Laplace
    - **sgd**: SGDClassifier otimizado com TF-IDF
    - **bayesian_network**: Rede Bayesiana (requer pgmpy)
    
    ## Como Funciona
    
    1. Os modelos são carregados de arquivos .pkl salvos
    2. Se um modelo não existir, ele será treinado automaticamente
    3. Use `/classify` para classificar textos
    4. Use `/classify/batch` para classificar múltiplos textos
    
    ## Treinamento
    
    Execute `python train_all_models.py` para treinar todos os modelos de uma vez.
    """,
    version="2.0.0"
)

# CORS
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
    

# ==================== GERENCIAMENTO DE MODELOS ====================

class ModelManager:
    """Gerencia o carregamento e uso de modelos salvos."""
    
    def __init__(self):
        self.naive_bayes: Optional[NaiveBayesClassifier] = None
        self.sgd: Optional[SGDSentimentClassifier] = None
        self.bayesian_network: Optional[BayesianNetworkClassifier] = None
        self.df: Optional[pd.DataFrame] = None
    
    def _get_model_path(self, classifier_type: ClassifierType) -> str:
        """Retorna o caminho do arquivo do modelo."""
        model_names = {
            ClassifierType.naive_bayes: "naive_bayes_model.pkl",
            ClassifierType.sgd: "sgd_model.pkl",
            ClassifierType.bayesian_network: "bayesian_network_model.pkl"
        }
        return os.path.join(MODELS_DIR, model_names[classifier_type])
    
    def _load_from_file(self, classifier_type: ClassifierType):
        """Carrega modelo de arquivo pickle."""
        model_path = self._get_model_path(classifier_type)
        
        if not os.path.exists(model_path):
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"✅ Modelo {classifier_type.value} carregado de {model_path}")
            return model
        except Exception as e:
            print(f"❌ Erro ao carregar modelo {classifier_type.value}: {e}")
            return None
    
    def _save_to_file(self, classifier_type: ClassifierType, model):
        """Salva modelo em arquivo pickle."""
        model_path = self._get_model_path(classifier_type)
        
        # Cria pasta se não existir
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"💾 Modelo {classifier_type.value} salvo em {model_path}")
        except Exception as e:
            print(f"❌ Erro ao salvar modelo {classifier_type.value}: {e}")
    
    def load_data(self):
        """Carrega os dados do CSV (apenas se necessário para treinar)."""
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
        """
        Retorna o classificador solicitado.
        Prioridade: 1) Já carregado na memória, 2) Carregar de arquivo, 3) Treinar novo
        """
        
        # 1. Verifica se já está na memória
        if classifier_type == ClassifierType.naive_bayes and self.naive_bayes is not None:
            return self.naive_bayes
        elif classifier_type == ClassifierType.sgd and self.sgd is not None:
            return self.sgd
        elif classifier_type == ClassifierType.bayesian_network and self.bayesian_network is not None:
            return self.bayesian_network
        
        # 2. Tenta carregar de arquivo
        model = self._load_from_file(classifier_type)
        
        # 3. Se não encontrou, treina novo
        if model is None:
            print(f"⚠️ Modelo {classifier_type.value} não encontrado. Treinando...")
            print(f"💡 Dica: Execute 'python train_all_models.py' para treinar todos de uma vez")
            model = self._train_new_model(classifier_type)
            if model is not None:
                self._save_to_file(classifier_type, model)
        
        # Armazena na memória
        if classifier_type == ClassifierType.naive_bayes:
            self.naive_bayes = model
        elif classifier_type == ClassifierType.sgd:
            self.sgd = model
        elif classifier_type == ClassifierType.bayesian_network:
            self.bayesian_network = model
        
        return model
    
    def _train_new_model(self, classifier_type: ClassifierType):
        """Treina um novo modelo (fallback se não encontrar arquivo)."""
        df = self.load_data()
        
        try:
            if classifier_type == ClassifierType.naive_bayes:
                print("🔄 Treinando Naive Bayes...")
                model = NaiveBayesClassifier(preprocessing="negation")
                df_train = df.copy()
                df_train['tokens'] = df_train['texto_completo'].apply(model._tokenize)
                model.train(df_train)
                print("✅ Naive Bayes pronto!")
                return model
            
            elif classifier_type == ClassifierType.sgd:
                print("🔄 Treinando SGD Classifier...")
                model = SGDSentimentClassifier()
                model.train(df)
                print("✅ SGD Classifier pronto!")
                return model
            
            elif classifier_type == ClassifierType.bayesian_network:
                print("🔄 Treinando Rede Bayesiana (isso pode demorar)...")
                model = BayesianNetworkClassifier()
                model.train(df)
                print("✅ Rede Bayesiana pronta!")
                return model
        except Exception as e:
            print(f"❌ Erro ao treinar {classifier_type.value}: {e}")
            return None


model_manager = ModelManager()


# ==================== EVENTOS ====================

@app.on_event("startup")
async def startup_event():
    """Carrega modelos salvos na inicialização."""
    print("\n🚀 Iniciando API de Análise de Sentimentos...")
    print(f"📁 Pasta de modelos: {MODELS_DIR}")
    
    # Tenta carregar Naive Bayes (modelo padrão)
    try:
        model_manager.get_classifier(ClassifierType.naive_bayes)
        print("✅ API pronta para receber requisições!\n")
    except Exception as e:
        print(f"⚠️ Aviso: {e}")
        print("💡 Execute 'python train_all_models.py' para treinar os modelos")
        print("✅ API iniciada, mas pode precisar treinar modelos na primeira requisição\n")


# ==================== ENDPOINTS ====================

@app.get("/", tags=["Info"])
async def root():
    """Endpoint raiz com informações da API."""
    return {
        "name": "API de Análise de Sentimentos",
        "version": "2.0.0",
        "status": "online",
        "docs": "/docs",
        "models_dir": MODELS_DIR
    }


@app.get("/health", tags=["Info"])
async def health_check():
    """Verifica o status da API e modelos."""
    models_status = {}
    
    for classifier_type in ClassifierType:
        model_path = model_manager._get_model_path(classifier_type)
        models_status[classifier_type.value] = {
            "loaded_in_memory": (
                (classifier_type == ClassifierType.naive_bayes and model_manager.naive_bayes is not None) or
                (classifier_type == ClassifierType.sgd and model_manager.sgd is not None) or
                (classifier_type == ClassifierType.bayesian_network and model_manager.bayesian_network is not None)
            ),
            "file_exists": os.path.exists(model_path),
            "file_path": model_path
        }
    
    return {
        "status": "healthy",
        "models": models_status
    }


@app.get("/models", response_model=ModelInfoResponse, tags=["Info"])
async def get_models_info():
    """Retorna informações sobre os modelos."""
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
        
        if classifier is None:
            raise HTTPException(
                status_code=503,
                detail=f"Modelo {request.classifier.value} não disponível. Execute 'python train_all_models.py'"
            )
        
        result = classifier.predict_with_details(request.text)
        
        return ClassifyResponse(
            text=request.text,
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            classifier_used=request.classifier.value
        )
    except HTTPException:
        raise
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
        
        if classifier is None:
            raise HTTPException(
                status_code=503,
                detail=f"Modelo {request.classifier.value} não disponível"
            )
        
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train/{classifier_type}", tags=["Treinamento"])
async def train_model(classifier_type: ClassifierType):
    """
    Força o (re)treinamento de um modelo específico e salva em arquivo.
    
    Útil para atualizar modelos com novos dados.
    """
    try:
        # Remove modelo da memória
        if classifier_type == ClassifierType.naive_bayes:
            model_manager.naive_bayes = None
        elif classifier_type == ClassifierType.sgd:
            model_manager.sgd = None
        elif classifier_type == ClassifierType.bayesian_network:
            model_manager.bayesian_network = None
        
        # Treina novo
        model = model_manager._train_new_model(classifier_type)
        
        if model is None:
            raise HTTPException(status_code=500, detail="Falha ao treinar modelo")
        
        # Salva em arquivo
        model_manager._save_to_file(classifier_type, model)
        
        # Atualiza na memória
        if classifier_type == ClassifierType.naive_bayes:
            model_manager.naive_bayes = model
        elif classifier_type == ClassifierType.sgd:
            model_manager.sgd = model
        elif classifier_type == ClassifierType.bayesian_network:
            model_manager.bayesian_network = model
        
        info = model.get_info()
        
        return {
            "message": f"Modelo {classifier_type.value} treinado e salvo com sucesso",
            "model_info": info,
            "saved_to": model_manager._get_model_path(classifier_type)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== EXECUÇÃO ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
