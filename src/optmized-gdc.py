import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

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

def train_sgd_classifier(df: pd.DataFrame) -> Pipeline:
    
    def categorize_rating(rating):
        if rating in [1, 2]: return "Negativa"
        if rating == 3: return "Neutra"
        if rating in [4, 5]: return "Positiva"
        return "Desconhecida"
        
    df['Sentiment_Target'] = df["overall_rating"].apply(categorize_rating)
    df = df[df['Sentiment_Target'] != 'Desconhecida'].copy()
    df['clean_review'] = df["complete_review"].apply(preprocess_text_for_vectorizer)
    
    X = df['clean_review']
    y = df['Sentiment_Target']
    
    # 1. CÁLCULO MANUAL DO PESO COM REFORÇO PARA NEUTRA
    le = LabelEncoder()
    y_ind = le.fit_transform(y)
    classes = le.classes_
    
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    
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
    model_pipeline.fit(X, y)
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

def main():
    parsed_data = parse_csv_data(file_path="data/dataset.csv") 

    optimized_model = train_sgd_classifier(parsed_data)

    print("INÍCIO DA CLASSIFICAÇÃO DE NOVOS TEXTOS")
    
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