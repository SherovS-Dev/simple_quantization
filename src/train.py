# src/train.py
import torch
import torch.nn as nn
from torch.optim import Adam
from src.data import get_dataloaders
from src.model import SentimentNet


def train_model(file_path, model_path, epochs=10, batch_size=32, max_samples=50000, max_features=5000):
    train_loader, test_loader, vectorizer = get_dataloaders(file_path, batch_size, max_samples, max_features)

    input_dim = max_features
    model = SentimentNet(input_dim)

    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            texts = batch['text']
            labels = batch['label'].unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    # Сохранение модели
    torch.save(model.state_dict(), model_path)
    print(f"Модель сохранена в {model_path}")