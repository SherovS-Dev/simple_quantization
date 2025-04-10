import torch
from src.model import SentimentNet  # Убедитесь, что путь к вашей модели правильный

def convert_to_onnx(pth_path, onnx_path, input_dim):
    # Загрузка модели
    model = SentimentNet(input_dim=input_dim)
    model.load_state_dict(torch.load(pth_path))
    model.eval()

    # Пример входных данных
    dummy_input = torch.randn(1, input_dim)

    # Экспорт в ONNX
    torch.onnx.export(model, dummy_input, onnx_path,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    print(f"Модель успешно конвертирована из {pth_path} в {onnx_path}")

if __name__ == "__main__":
    pth_path = "my_model/sentiment_model.pth"  # Путь к вашей .pth модели
    onnx_path = "my_model/sentiment_model.onnx"  # Путь для сохранения .onnx модели
    input_dim = 5000  # Укажите размерность входных данных вашей модели
    convert_to_onnx(pth_path, onnx_path, input_dim)