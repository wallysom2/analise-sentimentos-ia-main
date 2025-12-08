"""
Script para executar a API de Análise de Sentimentos
"""
import uvicorn
import sys
import os

# Adiciona o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 INICIANDO API DE ANÁLISE DE SENTIMENTOS")
    print("="*60)
    print("\n📍 Documentação disponível em: http://localhost:8000/docs")
    print("📍 Redoc em: http://localhost:8000/redoc\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Hot reload durante desenvolvimento
    )
