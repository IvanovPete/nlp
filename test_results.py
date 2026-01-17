import json
import sys
from typing import Dict, List, Any, Tuple


def check_endpoint_error(endpoint: str, response: Any) -> List[Tuple[str, str]]:
    """
    Проверяет ответ эндпоинта на наличие ошибок.
    Возвращает список кортежей (тип_ошибки, описание).
    """
    errors = []
    
    # Проверка на ошибку на уровне всего эндпоинта
    if isinstance(response, dict) and "error" in response:
        errors.append(("endpoint_error", f"Эндпоинт {endpoint} вернул ошибку: {response['error']}"))
        return errors
    
    # Проверка для NLTK эндпоинтов (имеют структуру {"results": [...]})
    if endpoint.startswith("/text_nltk/"):
        if not isinstance(response, dict):
            errors.append(("invalid_structure", f"Эндпоинт {endpoint}: ответ должен быть словарем"))
            return errors
        
        if "results" not in response:
            errors.append(("missing_field", f"Эндпоинт {endpoint}: отсутствует поле 'results'"))
            return errors
        
        if not isinstance(response["results"], list):
            errors.append(("invalid_type", f"Эндпоинт {endpoint}: поле 'results' должно быть списком"))
            return errors
        
        # Проверяем каждый результат в массиве
        for idx, result in enumerate(response["results"]):
            if isinstance(result, dict) and "error" in result:
                text_preview = result.get("text", "N/A")[:100] if "text" in result else "N/A"
                errors.append((
                    "item_error",
                    f"Эндпоинт {endpoint}, элемент {idx}: ошибка в тексте '{text_preview}...': {result['error'][:200]}"
                ))
    
    # Проверка для остальных эндпоинтов (tf-idf, bag-of-words, lsa, word2vec)
    else:
        if not isinstance(response, dict):
            errors.append(("invalid_structure", f"Эндпоинт {endpoint}: ответ должен быть словарем"))
            return errors
        
        # Проверяем наличие обязательных полей в зависимости от эндпоинта
        required_fields = {
            "/tf-idf": ["vocabulary", "tfidf_matrix", "shape"],
            "/bag-of-words": ["vocabulary", "bow_matrix", "shape"],
            "/lsa": ["lsa_matrix", "components", "explained_variance", "shape"],
            "/word2vec": ["vocabulary", "word_vectors", "shape"]
        }
        
        if endpoint in required_fields:
            for field in required_fields[endpoint]:
                if field not in response:
                    errors.append(("missing_field", f"Эндпоинт {endpoint}: отсутствует обязательное поле '{field}'"))
        
        # Проверяем корректность типов данных
        if "vocabulary" in response and not isinstance(response["vocabulary"], list):
            errors.append(("invalid_type", f"Эндпоинт {endpoint}: поле 'vocabulary' должно быть списком"))
        
        if "shape" in response:
            if not isinstance(response["shape"], list) or len(response["shape"]) != 2:
                errors.append(("invalid_type", f"Эндпоинт {endpoint}: поле 'shape' должно быть списком из 2 элементов"))
        
        # Проверяем на пустые результаты
        if "vocabulary" in response and len(response["vocabulary"]) == 0:
            errors.append(("empty_result", f"Эндпоинт {endpoint}: словарь пуст"))
        
        if "tfidf_matrix" in response:
            if not isinstance(response["tfidf_matrix"], list):
                errors.append(("invalid_type", f"Эндпоинт {endpoint}: поле 'tfidf_matrix' должно быть списком"))
            elif len(response["tfidf_matrix"]) == 0:
                errors.append(("empty_result", f"Эндпоинт {endpoint}: матрица TF-IDF пуста"))
        
        if "bow_matrix" in response:
            if not isinstance(response["bow_matrix"], list):
                errors.append(("invalid_type", f"Эндпоинт {endpoint}: поле 'bow_matrix' должно быть списком"))
            elif len(response["bow_matrix"]) == 0:
                errors.append(("empty_result", f"Эндпоинт {endpoint}: матрица Bag-of-Words пуста"))
        
        if "lsa_matrix" in response:
            if not isinstance(response["lsa_matrix"], list):
                errors.append(("invalid_type", f"Эндпоинт {endpoint}: поле 'lsa_matrix' должно быть списком"))
            elif len(response["lsa_matrix"]) == 0:
                errors.append(("empty_result", f"Эндпоинт {endpoint}: матрица LSA пуста"))
        
        if "word_vectors" in response:
            if not isinstance(response["word_vectors"], list):
                errors.append(("invalid_type", f"Эндпоинт {endpoint}: поле 'word_vectors' должно быть списком"))
            elif len(response["word_vectors"]) == 0:
                errors.append(("empty_result", f"Эндпоинт {endpoint}: векторы слов пусты"))
    
    return errors


def test_results_file(filename: str) -> Dict[str, Any]:
    """
    Тестирует файл с результатами на наличие ошибок.
    Возвращает статистику проверки.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ОШИБКА: Файл {filename} не найден")
        return {"total_endpoints": 0, "endpoints_with_errors": 0, "total_errors": 0}
    except json.JSONDecodeError as e:
        print(f"ОШИБКА: Не удалось распарсить JSON файл {filename}: {e}")
        return {"total_endpoints": 0, "endpoints_with_errors": 0, "total_errors": 0}
    
    all_errors = {}
    total_errors = 0
    
    print(f"Проверка файла: {filename}")
    print("=" * 80)
    
    for endpoint, response in data.items():
        errors = check_endpoint_error(endpoint, response)
        if errors:
            all_errors[endpoint] = errors
            total_errors += len(errors)
            print(f"\n[ERROR] {endpoint}: найдено {len(errors)} ошибок")
            for error_type, error_msg in errors:
                print(f"   [{error_type}] {error_msg}")
        else:
            print(f"[OK] {endpoint}: ошибок не найдено")
    
    print("\n" + "=" * 80)
    print(f"\nСтатистика:")
    print(f"  Всего эндпоинтов: {len(data)}")
    print(f"  Эндпоинтов с ошибками: {len(all_errors)}")
    print(f"  Всего ошибок: {total_errors}")
    
    if total_errors > 0:
        print(f"\n[WARNING] Обнаружены ошибки в {len(all_errors)} эндпоинтах!")
        return {
            "total_endpoints": len(data),
            "endpoints_with_errors": len(all_errors),
            "total_errors": total_errors,
            "errors_by_endpoint": all_errors,
            "has_errors": True
        }
    else:
        print(f"\n[SUCCESS] Все проверки пройдены успешно!")
        return {
            "total_endpoints": len(data),
            "endpoints_with_errors": 0,
            "total_errors": 0,
            "has_errors": False
        }


def main():
    """Главная функция для запуска тестов"""
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        # Ищем последний файл с результатами
        import glob
        import os
        result_files = glob.glob("results_*.json")
        if not result_files:
            print("ОШИБКА: Не найден файл с результатами")
            print("Использование: python test_results.py [имя_файла.json]")
            sys.exit(1)
        
        # Берем самый новый файл
        filename = max(result_files, key=os.path.getmtime)
        print(f"Используется файл: {filename}\n")
    
    result = test_results_file(filename)
    
    # Возвращаем код выхода в зависимости от наличия ошибок
    sys.exit(1 if result.get("has_errors", False) else 0)


if __name__ == "__main__":
    main()
