import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

MAX_FEATURES = 15000
NGRAM_RANGE = (1, 3)
portuguese_stopwords = set([
    'a', 'o', 'e', 'é', 'de', 'do', 'da', 'em', 'um', 'uma', 'os', 'as', 'que', 'para',
    'com', 'se', 'por', 'no', 'na', 'ao', 'aos', 'meu', 'minha', 'nosso', 'nossa', 'ele', 
    'ela', 'você', 'dele', 'dela', 'isso', 'isto', 'aquele', 'aquela', 'mas', 'ou', 'já', 
    'também', 'ser', 'ter', 'foi', 'pelo', 'pela', 'só', 'mais', 'menos', 
    'muito', 'pouco', 'nada', 'sem'
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

def parse_csv_data(file_path="data/base-reviews-b2w.csv"):
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

def prepare_data_for_training(df: pd.DataFrame):
    def categorize_rating(rating):
        if rating in [1, 2]: return "Negativa"
        if rating == 3: return "Neutra"
        if rating in [4, 5]: return "Positiva"
        return "Desconhecida"
        
    df['Sentiment_Target'] = df["overall_rating"].apply(categorize_rating)
    df = df[df['Sentiment_Target'] != 'Desconhecida'].copy()
    
    print("Processando textos (limpeza)...")
    df['clean_review'] = df["complete_review"].apply(preprocess_text_for_vectorizer)
    
    return df['clean_review'], df['Sentiment_Target']

def train_sgd_classifier(X_train, y_train) -> Pipeline:
    
    le = LabelEncoder()
    y_ind = le.fit_transform(y_train)
    classes = le.classes_
    
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    
    if 'Neutra' in classes:
        neutral_index = np.where(classes == 'Neutra')[0][0]
        weights[neutral_index] *= 2.5
    
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
    
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=NGRAM_RANGE,
            max_features=MAX_FEATURES
        )),
        ('clf', sgd_clf) 
    ])
    
    print("\n[INFO] Iniciando treinamento do SGDClassifier (Log Loss, Peso Reforçado)...")
    model_pipeline.fit(X_train, y_train)
    print(f"[SUCCESS] Treinamento concluído. Features (N-grams) usadas: {len(model_pipeline['tfidf'].get_feature_names_out())}")
    
    return model_pipeline

def classify_optimized_text(model_pipeline: Pipeline, new_review: str):
    
    processed_text = preprocess_text_for_vectorizer(new_review)
    
    predicted_sentiment = model_pipeline.predict([processed_text])[0]
    proba = model_pipeline.predict_proba([processed_text])[0]
    classes = model_pipeline.classes_
    
    highest_prob = np.max(proba)
    proba_details = dict(zip(classes, proba.round(4)))
    
    print("\n--- CLASSIFICAÇÃO DE NOVO TEXTO (SGDClassifier Otimizado) ---")
    print(f"TEXTO: '{new_review}'")
    print("------------------------------------")
    print(f"SENTIMENTO PREDIZIDO: **{predicted_sentiment}**")
    print(f"PROBABILIDADE MÁXIMA: {highest_prob:.2%}")
    print(f"PROBABILIDADES DETALHADAS: {proba_details}")

    return predicted_sentiment, highest_prob

def gerar_matriz_confusao(y_test, y_pred, labels, acc):
    output_dir = 'assets'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=labels, yticklabels=labels)
    
    plt.title(f'Matriz de Confusão - SGDClassifier - Acc {acc:.2%}')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'matriz_sgd_classifier.png')
    plt.savefig(save_path)
    print(f"\n✅ Matriz de confusão salva em: {save_path}")
    plt.show()

def main():
    try:
        parsed_data = parse_csv_data(file_path="data/base-reviews-b2w.csv") 
    except FileNotFoundError:
        print("Erro: Arquivo não encontrado.")
        return

    X, y = prepare_data_for_training(parsed_data)

    print(f"\nDividindo dados: Total {len(X)} linhas.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

    optimized_model = train_sgd_classifier(X_train, y_train)

    print("\n--- Avaliando no Conjunto de Teste ---")
    y_pred = optimized_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia SGD: {acc:.2%}")
    
    labels = ["Negativa", "Neutra", "Positiva"]
    gerar_matriz_confusao(y_test, y_pred, labels, acc)

    print("\nINÍCIO DA CLASSIFICAÇÃO DE NOVOS TEXTOS (MANUAL)")
    
    new_text_1 = "Produto triste."
    classify_optimized_text(optimized_model, new_text_1)

    new_text_2 = "Apesar da embalagem estar amassada, o produto é excelente e a qualidade é fantástica."
    classify_optimized_text(optimized_model, new_text_2)

    new_text_3 = "Eu não gostei do preço, mas o desempenho é bom."
    classify_optimized_text(optimized_model, new_text_3)

    new_text_4 = "Recebi a caixa. O item está de acordo com a foto no site."
    classify_optimized_text(optimized_model, new_text_4)

if __name__ == "__main__": 
    main()