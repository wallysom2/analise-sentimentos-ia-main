#!/bin/bash

# Inicia a API em background na porta 8000
echo "🚀 Iniciando API..."
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 &

# Espera alguns segundos para a API subir
sleep 5

# Inicia o Streamlit na porta que o Railway der ($PORT)
echo "🚀 Iniciando Frontend..."
python -m streamlit run frontend_streamlit.py --server.port $PORT --server.address 0.0.0.0
