import onnxruntime as ort
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def load_vectorizer(file_path, max_samples=50000, max_features=5000):
    df = pd.read_csv(file_path)
    df = df.sample(n=max_samples, random_state=42)
    texts = df['review'].tolist()
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectorizer.fit(texts)
    return vectorizer

def predict_sentiment_onnx(onnx_path, vectorizer, text):
    text_vector = vectorizer.transform([text]).toarray().astype(np.float32)
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    ort_inputs = {input_name: text_vector}
    ort_outs = ort_session.run([output_name], ort_inputs)
    prediction = ort_outs[0][0][0]
    return "Положительный" if prediction > 0.5 else "Отрицательный"

if __name__ == "__main__":
    file_path = "my_model/IMDB Dataset.csv"  # Путь к вашему датасету
    onnx_path = input("Введите путь к модели (например, my_model/sentiment_model.onnx): ")  # Путь к .onnx модели
    vectorizer = load_vectorizer(file_path)
    while True:
        text = input("Введите текст для анализа (или 'exit' для выхода): ")
        if text.lower() == 'exit':
            break
        sentiment = predict_sentiment_onnx(onnx_path, vectorizer, text)
        print(f"Предсказание: {sentiment}")