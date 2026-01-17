import requests
import json
from datetime import datetime

# URL сервера
BASE_URL = "http://localhost:8000"

# Чтение текстов из файла или использование предопределенных текстов
def load_texts_from_file(filename="corpus.txt"):
    """Загружает тексты из файла"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            # Читаем файл и разбиваем на тексты по пустым строкам или используем весь файл как один текст
            content = f.read()
            texts = [text.strip() for text in content.split('\n\n') if text.strip()]
            if not texts:
                # Если нет разделения по пустым строкам, используем весь файл как один текст
                texts = [content]
        return texts
    except FileNotFoundError:
        print(f"Файл {filename} не найден. Используются примеры текстов.")
        return None

def check_server():
    """Проверяет доступность сервера"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def send_request(endpoint, texts):
    """Отправляет запрос на указанный эндпоинт"""
    url = f"{BASE_URL}{endpoint}"
    payload = {"texts": texts}
    
    try:
        # Увеличиваем timeout для NER, так как он обрабатывает много текстов
        timeout = 600 if endpoint == "/text_nltk/ner" else 30
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к {endpoint}: {e}")
        return None

def main():
    # Проверяем доступность сервера
    print("Проверка доступности сервера...")
    if not check_server():
        print(f"ОШИБКА: Сервер недоступен по адресу {BASE_URL}")
        print("Убедитесь, что сервер запущен командой: uvicorn app:app --reload")
        return
    
    print("Сервер доступен!")
    print("-" * 50)
    
    # Загружаем тексты
    texts = load_texts_from_file()
    if texts is None:
        print("Тексты не загружены")
        return
    else:
        print(f"Загружено {len(texts)} текстов из файла corpus.txt")
    
    print("-" * 50)
    
    # Тестируем все эндпоинты
    endpoints = [
        "/tf-idf",
        "/bag-of-words",
        "/lsa",
        "/word2vec",
        "/text_nltk/tokenize",
        "/text_nltk/stem",
        "/text_nltk/lemmatize",
        "/text_nltk/pos",
        "/text_nltk/ner"
    ]
    
    # Создаем имя файла с временной меткой
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results_{timestamp}.json"
    
    # Словарь для хранения всех результатов
    all_results = {}
    
    for endpoint in endpoints:
        result = send_request(endpoint, texts)
        if result:
            # Сохраняем результат в словарь (без вывода в терминал)
            all_results[endpoint] = result
        else:
            # Выводим только ошибки
            print(f"Ошибка при запросе к {endpoint}")
            all_results[endpoint] = {"error": "Ошибка при запросе"}
    
    # Записываем все результаты в файл
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Результаты сохранены в файл: {results_file}")
    except Exception as e:
        print(f"Ошибка при сохранении результатов: {e}")

if __name__ == "__main__":
    main()

