# main.py
from src.train import train_model
from predict import predict_sentiment, load_vectorizer

if __name__ == "__main__":
    file_path = "my_model/IMDB Dataset.csv"
    model_path = "my_model/sentiment_model.pth"

    # Обучение модели
    train_model(file_path, model_path, epochs=10, batch_size=32, max_samples=50000, max_features=5000)

    # Пример использования
    vectorizer = load_vectorizer(file_path)
    text = "Фильм просто восторг!"
    sentiment = predict_sentiment(model_path, vectorizer, text)
    print(f"Предсказание для '{text}': {sentiment}")