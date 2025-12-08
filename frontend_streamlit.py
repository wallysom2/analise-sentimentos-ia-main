"""
Frontend Streamlit para Análise de Sentimentos
Interface visual para a API de classificação de sentimentos
"""
import streamlit as st
import requests
import json
from typing import Dict, Any

# ==================== CONFIGURAÇÃO ====================

API_URL = "http://localhost:8000"  # URL da API FastAPI

# ==================== FUNÇÕES ====================

def analisar_sentimento(texto: str, classificador: str = "naive_bayes") -> Dict[str, Any]:
    """
    Envia texto para a API e retorna o resultado da análise
    """
    try:
        response = requests.post(
            f"{API_URL}/classify",
            json={
                "text": texto,
                "classifier": classificador
            },
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Não foi possível conectar à API. Certifique-se de que ela está rodando!")
        st.info("Execute: `python run_api.py` ou `uvicorn src.api:app --reload`")
        return None
    except requests.exceptions.Timeout:
        st.error("A requisição demorou muito tempo. Tente novamente.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erro na requisição: {str(e)}")
        return None

def analisar_batch(textos: list, classificador: str = "naive_bayes") -> Dict[str, Any]:
    """
    Envia múltiplos textos para a API
    """
    try:
        response = requests.post(
            f"{API_URL}/classify/batch",
            json={
                "texts": textos,
                "classifier": classificador
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erro na requisição: {str(e)}")
        return None

def analisar_todos_modelos(texto: str) -> Dict[str, Any]:
    """
    Analisa o texto com TODOS os classificadores disponíveis
    Retorna um dicionário com os resultados de cada um
    """
    classificadores = ["naive_bayes", "sgd", "bayesian_network"]
    resultados = {}
    
    for clf in classificadores:
        try:
            response = requests.post(
                f"{API_URL}/classify",
                json={
                    "text": texto,
                    "classifier": clf
                },
                timeout=15
            )
            response.raise_for_status()
            resultados[clf] = response.json()
        except requests.exceptions.RequestException as e:
            resultados[clf] = {"error": str(e)}
    
    return resultados

# ==================== INTERFACE ====================

def main():
    # Configuração da página
    st.set_page_config(
        page_title="Análise de Sentimentos",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Título e descrição
    st.title("Análise de Sentimentos com IA")
    st.markdown("""
    Analise o sentimento de textos em português usando Machine Learning!
    
    **Escolha um classificador e digite seu texto para começar.**
    """)

    # Barra lateral - Configurações
    with st.sidebar:
        st.header("Configurações")
        
        classificador = st.selectbox(
            "Escolha o Classificador",
            ["naive_bayes", "sgd", "bayesian_network"],
            help="Selecione o algoritmo de classificação"
        )
        
        st.markdown("---")
        st.markdown("### Sobre os Classificadores")
        st.markdown("""
        - **Naive Bayes**: Simples e rápido, baseado em probabilidades
        - **SGD**: Otimizado com TF-IDF, boa performance
        - **Bayesian Network**: Rede probabilística avançada
        """)
        
        st.markdown("---")
        st.markdown("### Status da API")
        try:
            response = requests.get(f"{API_URL}/", timeout=2)
            if response.status_code == 200:
                st.success("API Online")
            else:
                st.error("API Offline")
        except:
            st.error("API Offline")
            st.info("Execute: `python run_api.py`")

    # Abas principais
    tab1, tab2, tab3 = st.tabs(["Análise Individual", "Comparar Modelos", "Análise em Lote"])

    # ==================== ABA 1: ANÁLISE INDIVIDUAL ====================
    with tab1:
        st.subheader("Analise um texto")
        
        # Input do usuário
        texto = st.text_area(
            "Digite ou cole seu texto aqui:",
            height=150,
            placeholder="Exemplo: Este produto é maravilhoso! Recomendo muito.",
            help="Digite o texto que deseja analisar"
        )
        
        # Botão de análise
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            analisar_btn = st.button("Analisar", type="primary", use_container_width=True)
        with col2:
            limpar_btn = st.button("Limpar", use_container_width=True)
        
        if limpar_btn:
            st.rerun()
        
        # Processar análise
        if analisar_btn:
            if not texto.strip():
                st.warning("Por favor, digite um texto para analisar!")
            else:
                with st.spinner("Analisando sentimento..."):
                    resultado = analisar_sentimento(texto, classificador)
                    
                    if resultado:
                        # Extrair informações
                        sentimento = resultado.get("sentiment", "desconhecido")
                        confianca = resultado.get("confidence", 0)
                        
                        # Exibir resultado em destaque
                        st.markdown("---")
                        st.markdown("### Resultado da Análise")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Sentimento Detectado")
                            st.markdown(f"**{sentimento.upper()}**")
                        
                        with col2:
                            st.metric(
                                "Confiança do Modelo",
                                f"{confianca:.2%}",
                                help="Quão confiante o modelo está na predição"
                            )
                            st.metric(
                                "Classificador Usado",
                                classificador.replace("_", " ").title()
                            )
                        
                        # Detalhes adicionais
                        with st.expander("Ver Detalhes Técnicos"):
                            st.json(resultado)

    # ==================== ABA 2: COMPARAR MODELOS ====================
    with tab2:
        st.subheader("Compare todos os modelos lado a lado")
        
        st.info("Veja como cada classificador analisa o mesmo texto")
        
        # Input do usuário
        texto_comparar = st.text_area(
            "Digite o texto para comparar:",
            height=120,
            placeholder="Digite um texto e veja como cada modelo o classifica...",
            key="texto_comparar"
        )
        
        if st.button("Comparar Todos os Modelos", type="primary", key="btn_comparar"):
            if not texto_comparar.strip():
                st.warning("Por favor, digite um texto para comparar!")
            else:
                with st.spinner("Analisando com todos os modelos..."):
                    resultados = analisar_todos_modelos(texto_comparar)
                    
                    if resultados:
                        st.markdown("---")
                        st.markdown("### Resultados Comparativos")
                        
                        # Exibir em 3 colunas lado a lado
                        col1, col2, col3 = st.columns(3)
                        
                        modelos_info = [
                            ("naive_bayes", "Naive Bayes", col1),
                            ("sgd", "SGD Classifier", col2),
                            ("bayesian_network", "Rede Bayesiana", col3)
                        ]
                        
                        for clf_key, clf_nome, coluna in modelos_info:
                            with coluna:
                                st.markdown(f"#### {clf_nome}")
                                
                                res = resultados.get(clf_key, {})
                                
                                if "error" in res:
                                    st.error(f"Erro: {res['error'][:50]}...")
                                else:
                                    sentimento = res.get("sentiment", "N/A")
                                    confianca = res.get("confidence", 0)
                                    
                                    # Sentimento
                                    st.markdown(f"**Sentimento:**")
                                    st.markdown(f"### {sentimento.upper()}")
                                    
                                    # Confiança
                                    st.metric("Confiança", f"{confianca:.2%}")
                                    
                                    # Probabilidades
                                    probs = res.get("probabilities", {})
                                    if probs:
                                        st.markdown("**Probabilidades:**")
                                        for sent, prob in probs.items():
                                            st.progress(prob, text=f"{sent}: {prob:.2%}")
                        
                        # Resumo comparativo
                        st.markdown("---")
                        st.markdown("### Resumo")
                        
                        # Tabela comparativa
                        dados_tabela = []
                        for clf_key, clf_nome, _ in modelos_info:
                            res = resultados.get(clf_key, {})
                            if "error" not in res:
                                dados_tabela.append({
                                    "Modelo": clf_nome,
                                    "Sentimento": res.get("sentiment", "N/A").upper(),
                                    "Confiança": f"{res.get('confidence', 0):.2%}"
                                })
                        
                        if dados_tabela:
                            st.dataframe(dados_tabela, use_container_width=True)
                        
                        # Verificar consenso
                        sentimentos = [resultados.get(clf, {}).get("sentiment") 
                                      for clf in ["naive_bayes", "sgd", "bayesian_network"]
                                      if "error" not in resultados.get(clf, {"error": True})]
                        
                        if sentimentos:
                            if len(set(sentimentos)) == 1:
                                st.success(f"Todos os modelos concordam: **{sentimentos[0].upper()}**")
                            else:
                                st.warning("Os modelos têm opiniões diferentes sobre este texto")
                        
                        # Detalhes técnicos
                        with st.expander("Ver Todos os Detalhes (JSON)"):
                            st.json(resultados)

    # ==================== ABA 3: ANÁLISE EM LOTE ====================
    with tab3:
        st.subheader("Analise múltiplos textos de uma vez")
        
        st.info("Digite um texto por linha")
        
        # Input de múltiplos textos
        textos_batch = st.text_area(
            "Digite seus textos (um por linha):",
            height=200,
            placeholder="Exemplo:\nEste produto é ótimo!\nNão gostei do atendimento.\nO serviço é normal.",
            help="Digite um texto por linha"
        )
        
        # Botão de análise em lote
        if st.button("Analisar Todos", type="primary"):
            if not textos_batch.strip():
                st.warning("Por favor, digite pelo menos um texto!")
            else:
                # Dividir textos por linha
                textos = [t.strip() for t in textos_batch.split("\n") if t.strip()]
                
                with st.spinner(f"Analisando {len(textos)} textos..."):
                    resultado = analisar_batch(textos, classificador)
                    
                    if resultado:
                        st.success(f"{len(textos)} textos analisados com sucesso!")
                        
                        # Exibir resultados em tabela
                        st.markdown("---")
                        st.markdown("### Resultados")
                        
                        # Criar dataframe para visualização
                        results_data = []
                        for i, res in enumerate(resultado.get("results", []), 1):
                            results_data.append({
                                "#": i,
                                "Texto": res.get("text", "")[:50] + "...",
                                "Sentimento": res.get('sentiment', '').upper(),
                                "Confiança": f"{res.get('confidence', 0):.2%}"
                            })
                        
                        st.dataframe(results_data, use_container_width=True)
                        
                        # Estatísticas
                        st.markdown("---")
                        st.markdown("### Estatísticas")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        sentimentos = [r.get("sentiment", "") for r in resultado.get("results", [])]
                        
                        with col1:
                            positivos = sentimentos.count("positivo")
                            st.metric("Positivos", f"{positivos} ({positivos/len(sentimentos)*100:.1f}%)")
                        
                        with col2:
                            negativos = sentimentos.count("negativo")
                            st.metric("Negativos", f"{negativos} ({negativos/len(sentimentos)*100:.1f}%)")
                        
                        with col3:
                            neutros = sentimentos.count("neutro")
                            st.metric("Neutros", f"{neutros} ({neutros/len(sentimentos)*100:.1f}%)")
                        
                        # Detalhes técnicos
                        with st.expander("Ver Resultados Completos (JSON)"):
                            st.json(resultado)

    # Rodapé
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Desenvolvido com Streamlit + FastAPI</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== EXECUÇÃO ====================

if __name__ == "__main__":
    main()
