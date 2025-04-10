# src/predict.py
import torch
from src.model import SentimentNet
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def load_vectorizer(file_path, max_samples=50000, max_features=5000):
    df = pd.read_csv(file_path)
    df = df.sample(n=max_samples, random_state=42)
    texts = df['review'].tolist()
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectorizer.fit(texts)
    return vectorizer


def predict_sentiment(model_path, vectorizer, text):
    model = SentimentNet(input_dim=5000)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    text_vectorized = vectorizer.transform([text]).toarray()
    text_tensor = torch.tensor(text_vectorized, dtype=torch.float32)

    with torch.no_grad():
        output = model(text_tensor)
        prediction = output.item()

    return "Положительный" if prediction > 0.5 else "Отрицательный"


if __name__ == "__main__":
    file_path = "my_model/IMDB Dataset.csv"
    model_path = input("Введите путь к модели (например, my_model/sentiment_model.pth): ")

    vectorizer = load_vectorizer(file_path)

    while True:
        text = input("Введите текст для анализа (или 'exit' для выхода): ")
        if text.lower() == 'exit':
            break
        sentiment = predict_sentiment(model_path, vectorizer, text)
        print(f"Предсказание: {sentiment}")