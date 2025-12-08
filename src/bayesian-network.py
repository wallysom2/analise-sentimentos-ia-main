import pandas as pd
import numpy as np
import re
import os 
import matplotlib.pyplot as plt
import seaborn as sns 


from pgmpy.estimators import HillClimbSearch, BayesianEstimator
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.inference.base import DiscreteFactor


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

N_TOP_WORDS = 30

portuguese_stopwords = set([
    'a', 'o', 'e', 'é', 'de', 'do', 'da', 'em', 'um', 'uma', 'os', 'as', 'que', 'para',
    'com', 'se', 'por', 'no', 'na', 'ao', 'aos', 'meu', 'minha', 'nosso', 'nossa', 'ele', 
    'ela', 'você', 'dele', 'dela', 'isso', 'isto', 'aquele', 'aquela', 'mas', 'ou', 'já', 
    'também', 'ser', 'ter', 'foi', 'pelo', 'pela', 'só', 'mais', 'menos'
])

def overall_rating_converter(rating_str):
    try:
        return int(float(rating_str))
    except (ValueError, TypeError):
        return None

def preprocess_text_for_vectorizer(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Záéíóúãõç\s]', ' ', text)
    tokens = text.split()
    return " ".join([w for w in tokens if w not in portuguese_stopwords and len(w) > 2])

def parse_csv_data(file_path="data/dataset.csv"):
    csv_data: pd.DataFrame = pd.read_csv(
        file_path, 
        low_memory=False, 
        usecols=["overall_rating", "review_title", "review_text"],
        converters={"overall_rating": overall_rating_converter}
    )
    
    csv_data["review_title"] = csv_data["review_title"].fillna("")
    csv_data["review_text"] = csv_data["review_text"].fillna("")
    csv_data["complete_review"] = csv_data["review_title"] + " " + csv_data["review_text"]
    csv_data = csv_data.dropna(subset=['overall_rating'])
    
    return csv_data.drop(columns=["review_title", "review_text"])

def preprocess_and_feature_engineer_tfidf(df: pd.DataFrame) -> tuple[pd.DataFrame, TfidfVectorizer, list[str]]:
    data = df.copy()

    def categorize_rating(rating):
        if rating in [1, 2]: return "Negativa"
        if rating == 3: return "Neutra"
        if rating in [4, 5]: return "Positiva"
        return "Desconhecida"
        
    data["Sentiment_Target"] = data["overall_rating"].apply(categorize_rating).astype('category')
    data = data.drop(columns=["overall_rating"])
    
    data["clean_review"] = data["complete_review"].apply(preprocess_text_for_vectorizer)

    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(data["clean_review"])
    
    feature_names = vectorizer.get_feature_names_out()
    sum_tfidf = tfidf_matrix.sum(axis=0)
    tfidf_scores = [(feature_names[i], sum_tfidf[0, i]) for i in range(tfidf_matrix.shape[1])]
    
    sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
    top_words = [word for word, score in sorted_scores[:N_TOP_WORDS]]
    
    print(f"\n--- Top {N_TOP_WORDS} Palavras (Geradas pelo TF-IDF) ---")
    print(top_words)

    for word in top_words:
        col_name = f"Palavra_{word}_presente"
        data[col_name] = data["clean_review"].apply(
            lambda text: "Sim" if word in text else "Não"
        ).astype('category')
        
    data = data.drop(columns=["complete_review", "clean_review"])
    
    return data, vectorizer, top_words

def train_bayesian_network(sample: pd.DataFrame) -> DiscreteBayesianNetwork:
    print("\nIniciando treinamento da Rede Bayesiana...")
    hc = HillClimbSearch(sample)
    
    best_model = hc.estimate(
        scoring_method="bdeu",
        max_indegree=4,
        epsilon=1e-4
    )
    
    model = DiscreteBayesianNetwork(best_model.edges())
    model.fit(sample, estimator=BayesianEstimator)
    print("Treinamento concluído.")
    
    return model

def classify_new_text(model, top_words, new_review: str):
    clean_review = preprocess_text_for_vectorizer(new_review)
    new_data = {}
    
    for word in top_words:
        col_name = f"Palavra_{word}_presente"
        new_data[col_name] = "Sim" if word in clean_review else "Não"
    
    if not new_data:
        print("\n--- Classificação Falhou ---")
        print("Nenhuma das top_words foi encontrada no novo texto.")
        return "Indefinido", 0.0
        
    evidence_df = pd.DataFrame([new_data])
    
    inference = VariableElimination(model)
    evidence = evidence_df.iloc[0].to_dict()
    
    result: DiscreteFactor = inference.query(
        variables=["Sentiment_Target"],
        evidence=evidence,
        show_progress=False
    )
    
    highest_prob = result.values.max()
    predicted_sentiment = result.state_names['Sentiment_Target'][np.argmax(result.values)]
    
    print("\n--- Resultados da Classificação do Novo Texto ---")
    print(f"Texto: '{new_review}'")
    print(f"Features Usadas: {evidence}")
    print(result)
    print(f"\nO Sentimento Mais Provável é: **{predicted_sentiment}** ({highest_prob:.2%})")
    
    return predicted_sentiment, highest_prob

def gerar_e_salvar_matriz_confusao(y_test, y_pred, labels, acc):
    output_dir = 'assets'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Matriz de Confusão - Rede Bayesiana - Acc {acc:.2%}')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    
    save_path = os.path.join(output_dir, 'matriz_confusao_bayesiana.png')
    plt.savefig(save_path)
    print(f"\n✅ Matriz de confusão salva em: {save_path}")
    plt.show()

def main():
    try:
        parsed_data = parse_csv_data(file_path="data/base-reviews-b2w.csv") 
    except FileNotFoundError:
        print("Arquivo não encontrado. Verifique o caminho.")
        return

    processed_data, vectorizer_trained, top_words_trained = \
        preprocess_and_feature_engineer_tfidf(parsed_data)
    
    print("\n--- Preparando Dados ---")
    full_sample = processed_data.sample(frac=0.1, random_state=42).copy()
    
    train_data, test_data = train_test_split(full_sample, test_size=0.2, random_state=42)
    
    print(f"Tamanho total da amostra usada: {len(full_sample)}")
    print(f"Tamanho Treino (80%): {len(train_data)}")
    print(f"Tamanho Teste (20%): {len(test_data)}")
    
    model = train_bayesian_network(train_data)
    
    print("\n--- Iniciando Validação no Conjunto de Teste ---")
    
    X_test = test_data.drop(columns=["Sentiment_Target"])
    y_test = test_data["Sentiment_Target"]
    
    y_pred_df = model.predict(X_test)
    y_pred = y_pred_df["Sentiment_Target"]
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia no conjunto de teste: {acc:.2%}")
    
    classes_possiveis = ["Negativa", "Neutra", "Positiva"]
    gerar_e_salvar_matriz_confusao(y_test, y_pred, classes_possiveis, acc)

    print("\n--- Testes Manuais ---")
    new_text_1 = "Ruim demais."
    classify_new_text(model, top_words_trained, new_text_1)

    new_text_2 = "O produto é excelente, recomendo a todos! A entrega foi muito rápida."
    classify_new_text(model, top_words_trained, new_text_2)

    new_text_3 = "O problema com a entrega é péssimo e o atendimento foi ruim."
    classify_new_text(model, top_words_trained, new_text_3)

    new_text_4 = "Recebi a caixa. O item está de acordo com a foto no site."
    classify_new_text(model, top_words_trained, new_text_4)


if __name__ == "__main__": 
    main()