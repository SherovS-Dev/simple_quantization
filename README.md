# Проект анализа и оптимизации нейросетей для анализа тональности

Этот проект предоставляет утилиты для анализа и оптимизации нейросетей, предназначенных для анализа тональности текстов (на примере набора данных IMDB). Основной файл `optimize_model.py` позволяет анализировать характеристики моделей в форматах `.pth` и `.onnx`, а также квантизовать их с использованием различных методов.

## Структура проекта
```
├── my_model/
│   └── IMDB Dataset.csv  # Пример набора данных
├── src/
│   ├── data.py       # Загрузка и предобработка данных IMDB
│   ├── model.py      # Определение модели SentimentNet
│   └── train.py      # Обучение модели SentimentNet
├── main.py # Пример обучения и предсказания
├── convert_to_onnx.py # Конвертация модели .pth в .onnx
├── convert_to_pth.py # Конвертация модели .onnx в .pth
├── optimize_model.py # Анализ и квантизация моделей
├── predict.py # Предсказание с использованием модели .pth
├── predict_with_onnx.py # Предсказание с использованием модели .onnx
├── requirements.txt # Зависимости проекта
└── README.md
```


### Описание файлов

- **`src/data.py`**:  
  Содержит класс `IMDBDataset` и функции для загрузки данных из CSV-файла, их векторизации с помощью `TfidfVectorizer` и создания `DataLoader` для обучения и тестирования.

- **`src/model.py`**:  
  Определяет класс `SentimentNet` — простую нейросеть для анализа тональности с двумя линейными слоями.

- **`src/train.py`**:  
  Реализует обучение модели `SentimentNet` на данных IMDB и сохранение обученной модели в файл `.pth`.

- **`optimize_model.py`**:  
  Основной скрипт для анализа и квантизации моделей. Поддерживает:  
  - Анализ характеристик моделей `.pth` и `.onnx` (веса, размеры, типы данных).  
  - Квантизацию `.pth` методами `dynamic`, `static`, `qat`.  
  - Квантизацию `.onnx` методами `dynamic`, `static`.

- **`predict.py`**:  
  Скрипт для предсказания тональности текста с использованием модели `.pth`.

- **`convert_to_onnx.py`**:  
  Конвертирует модель `.pth` в формат `.onnx`.

- **`convert_to_pth.py`**:  
  Конвертирует модель `.onnx` в формат `.pth`.

- **`predict_onnx.py`**:  
  Скрипт для предсказания тональности текста с использованием модели `.onnx`.

- **`requirements.txt`**:  
  Содержит список зависимостей проекта.

- **`main.py`**:  
  Пример использования: обучение модели и предсказание.

- **`my_model/IMDB Dataset.csv`**:  
  Ожидаемый файл с данными (не включен). Должен содержать колонки `review` (текст отзыва) и `sentiment` (метка: `positive` или `negative`).

## Установка

1. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## Использование

### 1. Обучение модели
Используйте `main.py` для обучения модели::
```bash
  python main.py
  ```

- Обучает модель и сохраняет ее в `my_model/sentiment_model.pth`.

### 2. Анализ модели

Используйте `optimize_model.py` для анализа характеристик модели:
 ```bash
   python optimize_model.py
   ```
- Пример ввода:
    ```text
    Введите тип модели (pth или onnx): pth
    Введите путь к модели: sentiment_model.pth
    Выберите действие (анализ или квантизация): анализ
    ```

### 3. Квантизация модели
Используйте optimize_model.py для квантизации:
- Для `.pth`:
    ```text
    Введите тип модели (pth или onnx): pth
    Введите путь к модели: my_model/sentiment_model.pth
    Выберите действие (анализ или квантизация): квантизация
    Введите размерность входных данных (например, 5000): 5000
    Выберите метод квантизации (dynamic, static, qat): dynamic
    Введите путь для сохранения квантизованной модели: my_model/quant_model.pth
    ```

- Для `.onnx`:
    ```text
    Введите тип модели (pth или onnx): onnx
    Введите путь к модели: sentiment_model.onnx
    Выберите действие (анализ или квантизация): квантизация
    Выберите метод квантизации (dynamic, static): static
    Введите путь для сохранения квантизованной модели: quant_model_with_static.onnx
    ```

### 4. Предсказание
- Для `.pth` используйте `predict.py`:
  ```bash
  python predict.py
  ```
  
- Для `.onnx` используйте `predict_onnx.py`:
  ```bash
  python predict_onnx.py
  ```
  
### 4. Конвертация моделей
- Конвертация `.pth` в `.onnx`:
  ```bash
  python convert_to_onnx.py
  ```
  
- Конвертация `.onnx` в `.pth`:
  ```bash
  python convert_to_pth.py
  ```
  
Примечания
- QAT для .pth: Требует обучающих данных и переобучения. Убедитесь, что IMDB Dataset.csv доступен.
- Ошибки: Если модель уже квантизована, скрипт сообщит об этом и выведет ее характеристики.
- Static для .onnx: Требует калибровочных данных. Имя входа модели должно быть 'input', иначе настройте ONNXCalibrationDataReader.