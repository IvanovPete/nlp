# NLP Микросервис на FastAPI

Простой микросервис для обработки текстов с использованием методов NLP.

## Установка

1. Создайте виртуальное окружение:
```bash
python -m venv venv
```

2. Активируйте виртуальное окружение:
- Windows: `venv\Scripts\activate`
- Linux/Mac: `source venv/bin/activate`

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Запуск сервера

```bash
uvicorn app:app --reload
```

Сервер будет доступен по адресу: http://localhost:8000

## Запуск клиента

В другом терминале:
```bash
python client.py
```

## API Эндпоинты

- `GET /` - Информация о сервисе
- `POST /tf-idf` - TF-IDF анализ (на numpy)
- `POST /bag-of-words` - Bag of Words (на numpy)
- `POST /lsa` - Latent Semantic Analysis (из sklearn)
- `POST /word2vec` - Word2Vec (из sklearn)
- `POST /text_nltk/tokenize` - Токенизация
- `POST /text_nltk/stem` - Стемминг
- `POST /text_nltk/lemmatize` - Лемматизация
- `POST /text_nltk/pos` - Part-of-Speech tagging
- `POST /text_nltk/ner` - Named Entity Recognition

## Формат запроса

Все POST эндпоинты принимают JSON:
```json
{
  "texts": ["текст 1", "текст 2", "текст 3"]
}
```

## Пример использования

Клиент автоматически читает тексты из файла `corpus.txt` или использует примеры текстов, если файл не найден.

