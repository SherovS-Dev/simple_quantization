# src/data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import Dataset, DataLoader


class IMDBDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': torch.tensor(self.texts[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def load_data(file_path, max_samples=10000):
    df = pd.read_csv(file_path)
    df = df.sample(n=max_samples, random_state=42)  # Берём 10,000 случайных записей
    texts = df['review'].tolist()
    labels = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).tolist()
    return texts, labels


def preprocess_data(texts, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    texts_vectorized = vectorizer.fit_transform(texts).toarray()
    return texts_vectorized, vectorizer


def get_dataloaders(file_path, batch_size=32, max_samples=50000, max_features=5000):
    texts, labels = load_data(file_path, max_samples)
    texts_vectorized, vectorizer = preprocess_data(texts, max_features)
    X_train, X_test, y_train, y_test = train_test_split(texts_vectorized, labels, test_size=0.2, random_state=42)

    train_dataset = IMDBDataset(X_train, y_train)
    test_dataset = IMDBDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, vectorizer