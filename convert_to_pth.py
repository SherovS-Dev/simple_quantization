import onnx
from onnx2pytorch import ConvertModel # todo
import torch

def convert_to_pth(onnx_path, pth_path):
    # Загрузка ONNX модели
    onnx_model = onnx.load(onnx_path)
    # Конвертация в PyTorch
    pytorch_model = ConvertModel(onnx_model)
    # Сохранение в .pth
    torch.save(pytorch_model.state_dict(), pth_path)
    print(f"Модель успешно конвертирована из {onnx_path} в {pth_path}")

if __name__ == "__main__":
    onnx_path = "my_model/sentiment_model.onnx"  # Путь к .onnx модели
    pth_path = "my_model/sentiment_model_from_onnx.pth"  # Путь для сохранения .pth модели
    convert_to_pth(onnx_path, pth_path)