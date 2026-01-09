from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.chunk import ne_chunk

# Скачиваем необходимые данные NLTK
# Список необходимых данных NLTK
nltk_data = [
    'punkt',  # для токенизации
    'averaged_perceptron_tagger',  # для POS-тегирования
    'maxent_ne_chunker',  # для NER
    'words',  # словарь для NER
    'wordnet'  # для лемматизации
]

# Загружаем данные, если они отсутствуют
for data in nltk_data:
    try:
        if data == 'punkt':
            nltk.data.find('tokenizers/punkt')
        elif data == 'averaged_perceptron_tagger':
            nltk.data.find('taggers/averaged_perceptron_tagger')
        elif data == 'maxent_ne_chunker':
            nltk.data.find('chunkers/maxent_ne_chunker')
        elif data == 'words':
            nltk.data.find('corpora/words')
        elif data == 'wordnet':
            nltk.data.find('corpora/wordnet')
    except LookupError:
        print(f"Загрузка данных NLTK: {data}")
        try:
            nltk.download(data, quiet=True)
        except Exception as e:
            print(f"Ошибка при загрузке {data}: {e}")

app = FastAPI()

class TextRequest(BaseModel):
    texts: List[str]


@app.get("/")
def root():
    return {"message": "NLP Microservice на FastAPI", "endpoints": ["/tf-idf", "/bag-of-words", "/lsa", "/word2vec", "/text_nltk/tokenize", "/text_nltk/stem", "/text_nltk/lemmatize", "/text_nltk/pos", "/text_nltk/ner"]}


@app.post("/tf-idf")
def tf_idf(request: TextRequest):
    """TF-IDF на numpy"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="Список текстов не может быть пустым")
    
    # Простая реализация TF-IDF на numpy
    # Токенизация
    documents = [text.lower().split() for text in request.texts]
    
    # Создаем словарь всех уникальных слов
    all_words = set()
    for doc in documents:
        all_words.update(doc)
    vocab = sorted(list(all_words))
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    
    # Вычисляем TF (Term Frequency)
    tf = np.zeros((len(documents), len(vocab)))
    for i, doc in enumerate(documents):
        for word in doc:
            if word in vocab_index:
                tf[i, vocab_index[word]] += 1
        # Нормализация
        if len(doc) > 0:
            tf[i] = tf[i] / len(doc)
    
    # Вычисляем IDF (Inverse Document Frequency)
    df = np.sum(tf > 0, axis=0)
    idf = np.log(len(documents) / (df + 1e-10)) + 1
    
    # TF-IDF = TF * IDF
    tfidf = tf * idf
    
    return {
        "vocabulary": vocab,
        "tfidf_matrix": tfidf.tolist(),
        "shape": list(tfidf.shape)
    }


@app.post("/bag-of-words")
def bag_of_words(request: TextRequest):
    """Bag of Words на numpy"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="Список текстов не может быть пустым")
    
    # Токенизация
    documents = [text.lower().split() for text in request.texts]
    
    # Создаем словарь всех уникальных слов
    all_words = set()
    for doc in documents:
        all_words.update(doc)
    vocab = sorted(list(all_words))
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    
    # Создаем матрицу Bag of Words
    bow_matrix = np.zeros((len(documents), len(vocab)), dtype=int)
    for i, doc in enumerate(documents):
        for word in doc:
            if word in vocab_index:
                bow_matrix[i, vocab_index[word]] += 1
    
    return {
        "vocabulary": vocab,
        "bow_matrix": bow_matrix.tolist(),
        "shape": list(bow_matrix.shape)
    }


@app.post("/lsa")
def lsa(request: TextRequest):
    """Latent Semantic Analysis из sklearn"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="Список текстов не может быть пустым")
    
    # Используем TfidfVectorizer для создания матрицы TF-IDF
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(request.texts)
    
    # Применяем SVD для LSA
    n_components = min(10, len(request.texts) - 1, tfidf_matrix.shape[1])
    if n_components < 1:
        n_components = 1
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_matrix = svd.fit_transform(tfidf_matrix)
    
    return {
        "lsa_matrix": lsa_matrix.tolist(),
        "components": svd.components_.tolist(),
        "explained_variance": svd.explained_variance_ratio_.tolist(),
        "shape": list(lsa_matrix.shape)
    }


@app.post("/word2vec")
def word2vec(request: TextRequest):
    """Word2Vec из sklearn (используем CountVectorizer как простую альтернативу)"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="Список текстов не может быть пустым")
    
    # Примечание: sklearn не имеет прямого Word2Vec, используем CountVectorizer
    # Для настоящего Word2Vec нужен gensim, но по требованию используем sklearn
    vectorizer = CountVectorizer(max_features=100)
    word_vectors = vectorizer.fit_transform(request.texts)
    
    return {
        "vocabulary": vectorizer.get_feature_names_out().tolist(),
        "word_vectors": word_vectors.toarray().tolist(),
        "shape": list(word_vectors.shape)
    }


@app.post("/text_nltk/tokenize")
def tokenize(request: TextRequest):
    """Токенизация текста"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="Список текстов не может быть пустым")
    
    try:
        result = []
        for text in request.texts:
            if not text or not isinstance(text, str):
                continue
            try:
                words = word_tokenize(text)
                sentences = sent_tokenize(text)
                result.append({
                    "text": text,
                    "words": words,
                    "sentences": sentences
                })
            except Exception as e:
                result.append({
                    "text": text,
                    "error": f"Ошибка обработки текста: {str(e)}"
                })
        
        return {"results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при токенизации: {str(e)}\n{traceback.format_exc()}")


@app.post("/text_nltk/stem")
def stem(request: TextRequest):
    """Стемминг текста"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="Список текстов не может быть пустым")
    
    try:
        stemmer = PorterStemmer()
        result = []
        for text in request.texts:
            if not text or not isinstance(text, str):
                continue
            try:
                words = word_tokenize(text)
                stemmed_words = [stemmer.stem(word) for word in words]
                result.append({
                    "original": words,
                    "stemmed": stemmed_words
                })
            except Exception as e:
                result.append({
                    "text": text,
                    "error": f"Ошибка обработки текста: {str(e)}"
                })
        
        return {"results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при стемминге: {str(e)}\n{traceback.format_exc()}")


@app.post("/text_nltk/lemmatize")
def lemmatize(request: TextRequest):
    """Лемматизация текста"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="Список текстов не может быть пустым")
    
    try:
        lemmatizer = WordNetLemmatizer()
        result = []
        for text in request.texts:
            if not text or not isinstance(text, str):
                continue
            try:
                words = word_tokenize(text)
                lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
                result.append({
                    "original": words,
                    "lemmatized": lemmatized_words
                })
            except Exception as e:
                result.append({
                    "text": text,
                    "error": f"Ошибка обработки текста: {str(e)}"
                })
        
        return {"results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при лемматизации: {str(e)}\n{traceback.format_exc()}")


@app.post("/text_nltk/pos")
def pos_tagging(request: TextRequest):
    """Part-of-Speech (POS) tagging"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="Список текстов не может быть пустым")
    
    try:
        result = []
        for text in request.texts:
            if not text or not isinstance(text, str):
                continue
            try:
                words = word_tokenize(text)
                pos_tags = pos_tag(words)
                result.append({
                    "words": words,
                    "pos_tags": [{"word": word, "tag": tag} for word, tag in pos_tags]
                })
            except Exception as e:
                result.append({
                    "text": text,
                    "error": f"Ошибка обработки текста: {str(e)}"
                })
        
        return {"results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при POS-тегировании: {str(e)}\n{traceback.format_exc()}")


@app.post("/text_nltk/ner")
def ner(request: TextRequest):
    """Named Entity Recognition"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="Список текстов не может быть пустым")
    
    try:
        result = []
        for text in request.texts:
            if not text or not isinstance(text, str):
                continue
            try:
                words = word_tokenize(text)
                pos_tags = pos_tag(words)
                tree = ne_chunk(pos_tags)
                
                entities = []
                for subtree in tree:
                    if isinstance(subtree, nltk.Tree):
                        entity = " ".join([token for token, pos in subtree.leaves()])
                        entities.append({"entity": entity, "label": subtree.label()})
                
                result.append({
                    "text": text,
                    "entities": entities
                })
            except Exception as e:
                result.append({
                    "text": text,
                    "error": f"Ошибка обработки текста: {str(e)}"
                })
        
        return {"results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при NER: {str(e)}\n{traceback.format_exc()}")

