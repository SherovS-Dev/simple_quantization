import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic, get_default_qconfig, prepare, convert, prepare_qat
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic as ort_quantize_dynamic, quantize_static, CalibrationDataReader, QuantType
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Функция для загрузки калибровочных данных для .pth
def load_calibration_data(file_path, vectorizer, num_samples=1000):
    df = pd.read_csv(file_path)
    df = df.sample(n=min(num_samples, len(df)), random_state=42)
    texts = df['review'].tolist()
    texts_vectorized = vectorizer.transform(texts).toarray()
    return [torch.tensor(vec, dtype=torch.float32).unsqueeze(0) for vec in texts_vectorized]

# Функция для загрузки калибровочных данных для .onnx
def load_onnx_calibration_data(file_path, vectorizer, num_samples=1000):
    df = pd.read_csv(file_path)
    df = df.sample(n=min(num_samples, len(df)), random_state=42)
    texts = df['review'].tolist()
    texts_vectorized = vectorizer.transform(texts).toarray()
    return [np.array(vec, dtype=np.float32).reshape(1, -1) for vec in texts_vectorized]

# Проверка квантизации для .pth
def is_pth_quantized(state_dict):
    for param in state_dict.values():
        if isinstance(param, torch.Tensor) and (param.dtype == torch.quint8 or param.dtype == torch.qint8):
            return True
        elif isinstance(param, tuple):
            for p in param:
                if isinstance(p, torch.Tensor) and (p.dtype == torch.quint8 or p.dtype == torch.qint8):
                    return True
    return False

# Проверка квантизации для .onnx
def is_onnx_quantized(model):
    for initializer in model.graph.initializer:
        dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[initializer.data_type]
        if dtype == np.int8:
            return True
    return False

# Анализ модели .pth
def get_pth_model_info(pth_path, verbose=True):
    try:
        state_dict = torch.load(pth_path)
        if verbose:
            print(f"Анализ модели: {pth_path}")
        total_params = 0
        is_quantized = False
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                if verbose:
                    print(f"Parameter: {name}, Shape: {param.shape}, Data Type: {param.dtype}")
                total_params += param.numel()
                if param.dtype in (torch.quint8, torch.qint8):
                    is_quantized = True
            elif isinstance(param, tuple):
                if verbose:
                    print(f"Parameter: {name} (packed parameters):")
                for i, p in enumerate(param):
                    if isinstance(p, torch.Tensor):
                        if verbose:
                            print(f"  Subparameter [{i}]: Shape: {p.shape}, Data Type: {p.dtype}")
                        total_params += p.numel()
                        if p.dtype in (torch.quint8, torch.qint8):
                            is_quantized = True
            else:
                if verbose:
                    print(f"Parameter: {name}, Type: {type(param)}")
        total_size_mb = sum(p.element_size() * p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor)) / (1024 ** 2)
        if verbose:
            print(f"Общее количество параметров: {total_params}")
            print(f"Общий размер модели: {total_size_mb:.2f} MB")
            print("Модель квантизована до INT8" if is_quantized else "Модель не квантизована")
        return is_quantized
    except FileNotFoundError:
        print(f"Ошибка: файл {pth_path} не найден")
        return None

# Анализ модели .onnx
def get_onnx_model_info(onnx_path, verbose=True):
    try:
        model = onnx.load(onnx_path)
        if verbose:
            print(f"Анализ модели: {onnx_path}")
        total_params = 0
        is_quantized = False
        for initializer in model.graph.initializer:
            tensor = initializer
            dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tensor.data_type]
            shape = [dim for dim in tensor.dims]
            if verbose:
                print(f"Initializer: {tensor.name}, Shape: {shape}, Data Type: {dtype}")
            param_size = np.prod(shape)
            total_params += param_size
            if dtype == np.int8:
                is_quantized = True
        total_size_mb = sum(np.prod([dim for dim in init.dims]) * onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[init.data_type].itemsize for init in model.graph.initializer) / (1024 ** 2)
        if verbose:
            print(f"Общее количество параметров: {total_params}")
            print(f"Общий размер модели: {total_size_mb:.2f} MB")
            print("Модель квантизована до INT8" if is_quantized else "Модель не квантизована")
        return is_quantized
    except FileNotFoundError:
        print(f"Ошибка: файл {onnx_path} не найден")
        return None

# Квантизация .pth
def quantize_pth(model, method, quantized_pth_path, calibration_data=None, train_loader=None):
    model.eval()
    if method == "dynamic":
        quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        torch.save(quantized_model.state_dict(), quantized_pth_path)
        print(f"Модель квантизована методом dynamic и сохранена в {quantized_pth_path}")
    elif method == "static":
        if not calibration_data:
            raise ValueError("Для static quantization нужны калибровочные данные")
        qconfig = get_default_qconfig('fbgemm')
        model.qconfig = qconfig
        model_prepared = prepare(model)
        for data in calibration_data:
            model_prepared(data)
        quantized_model = convert(model_prepared)
        torch.save(quantized_model.state_dict(), quantized_pth_path)
        print(f"Модель квантизована методом static и сохранена в {quantized_pth_path}")
    elif method == "qat":
        if not train_loader:
            raise ValueError("Для QAT quantization нужен train_loader для переобучения модели")
        model.qconfig = get_default_qconfig('fbgemm')
        model_qat = prepare_qat(model, inplace=False)
        model_qat.train()
        optimizer = torch.optim.Adam(model_qat.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()
        for epoch in range(3):
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model_qat(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            print(f"Эпоха {epoch + 1} завершена")
        model_qat.eval()
        quantized_model = convert(model_qat)
        torch.save(quantized_model.state_dict(), quantized_pth_path)
        print(f"Модель квантизована методом QAT и сохранена в {quantized_pth_path}")
    else:
        raise ValueError("Метод должен быть: dynamic, static, qat")

# Квантизация .onnx
def quantize_onnx(onnx_path, method, quantized_onnx_path, calibration_data=None):
    if method == "dynamic":
        ort_quantize_dynamic(onnx_path, quantized_onnx_path, weight_type=ort.QuantType.QInt8)
        print(f"Модель квантизована методом dynamic и сохранена в {quantized_onnx_path}")
    elif method == "static":
        if not calibration_data:
            raise ValueError("Для static quantization нужны калибровочные данные")
        class ONNXCalibrationDataReader(CalibrationDataReader):
            def __init__(self, calibration_data):
                self.data = iter(calibration_data)
            def get_next(self):
                try:
                    data = next(self.data)
                    return {"input": data}
                except StopIteration:
                    return None
        data_reader = ONNXCalibrationDataReader(calibration_data)
        quantize_static(
            model_input=onnx_path,
            model_output=quantized_onnx_path,
            calibration_data_reader=data_reader,
            quant_format=ort.quantization.QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8
        )
        print(f"Модель квантизована методом static и сохранена в {quantized_onnx_path}")
    else:
        raise ValueError("Метод должен быть: dynamic, static")

# Класс модели
class SentimentNet(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(SentimentNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Основной блок
if __name__ == "__main__":
    print("Добро пожаловать в утилиту для анализа и оптимизации моделей!")
    model_type = input("Введите тип модели (pth или onnx): ").strip().lower()
    model_path = input("Введите путь к модели: ").strip()

    if model_type == "pth":
        action = input("Выберите действие (анализ или квантизация): ").strip().lower()
        if action == "анализ":
            get_pth_model_info(model_path)
        elif action == "квантизация":
            state_dict = torch.load(model_path)
            is_quantized = get_pth_model_info(model_path, verbose=False)
            if is_quantized:
                print("Модель уже квантизована до INT8")
                get_pth_model_info(model_path)
            else:
                input_dim = int(input("Введите размерность входных данных (например, 5000): ").strip())
                output_dim = state_dict['fc2.weight'].shape[0]
                model = SentimentNet(input_dim, output_dim)
                try:
                    model.load_state_dict(state_dict)
                except RuntimeError as e:
                    print(f"Ошибка загрузки модели: {e}")
                    exit(1)

                df = pd.read_csv("my_model/IMDB Dataset.csv")
                vectorizer = TfidfVectorizer(max_features=input_dim)
                vectorizer.fit(df['review'].tolist())

                method = input("Выберите метод квантизации (dynamic, static, qat): ").strip().lower()
                quantized_pth_path = input("Введите путь для сохранения квантизованной модели: ").strip()

                if method in ["static", "qat"]:
                    calibration_data = load_calibration_data("my_model/IMDB Dataset.csv", vectorizer)
                    if method == "qat":
                        train_loader = None  # Замените на реальный DataLoader
                        quantize_pth(model, method, quantized_pth_path, calibration_data, train_loader)
                    else:
                        quantize_pth(model, method, quantized_pth_path, calibration_data)
                else:
                    quantize_pth(model, method, quantized_pth_path)
    elif model_type == "onnx":
        action = input("Выберите действие (анализ или квантизация): ").strip().lower()
        if action == "анализ":
            get_onnx_model_info(model_path)
        elif action == "квантизация":
            model = onnx.load(model_path)
            is_quantized = get_onnx_model_info(model_path, verbose=False)
            if is_quantized:
                print("Модель уже квантизована до INT8")
                get_onnx_model_info(model_path)
            else:
                method = input("Выберите метод квантизации (dynamic, static): ").strip().lower()
                quantized_onnx_path = input("Введите путь для сохранения квантизованной модели: ").strip()
                if method == "static":
                    df = pd.read_csv("my_model/IMDB Dataset.csv")
                    vectorizer = TfidfVectorizer(max_features=5000)  # Замените на ваш max_features
                    vectorizer.fit(df['review'].tolist())
                    calibration_data = load_onnx_calibration_data("my_model/IMDB Dataset.csv", vectorizer)
                    quantize_onnx(model_path, method, quantized_onnx_path, calibration_data)
                else:
                    quantize_onnx(model_path, method, quantized_onnx_path)
    else:
        print("Ошибка: тип модели должен быть 'pth' или 'onnx'")