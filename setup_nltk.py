"""
Скрипт для проверки и загрузки всех необходимых NLTK ресурсов
"""
import nltk

# Список необходимых данных NLTK с путями для проверки
nltk_resources = [
    ('punkt', 'tokenizers/punkt'),
    ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
    # для английского языка
    ('averaged_perceptron_tagger_eng',
     'taggers/averaged_perceptron_tagger_eng'),
    ('maxent_ne_chunker', 'chunkers/maxent_ne_chunker'),
    ('maxent_ne_chunker_tab', 'chunkers/maxent_ne_chunker_tab'),
    ('words', 'corpora/words'),
    ('wordnet', 'corpora/wordnet')
]

print("Проверка и загрузка NLTK ресурсов...")
print("=" * 60)

all_loaded = True

# Загружаем данные, если они отсутствуют
for resource_name, resource_path in nltk_resources:
    try:
        nltk.data.find(resource_path)
        print(f"[OK] {resource_name} - уже установлен")
    except LookupError:
        print(f"[LOADING] {resource_name} - загрузка...")
        try:
            nltk.download(resource_name, quiet=True)
            # Проверяем, что ресурс действительно загружен
            try:
                nltk.data.find(resource_path)
                print(f"[OK] {resource_name} - успешно загружен")
            except LookupError:
                print(f"[ERROR] {resource_name} - не найден после загрузки")
                all_loaded = False
        except Exception as e:
            print(f"[ERROR] {resource_name} - ошибка: {e}")
            all_loaded = False

print("=" * 60)
if all_loaded:
    print("Все NLTK ресурсы готовы к использованию!")
else:
    print("Некоторые ресурсы не удалось загрузить.")
