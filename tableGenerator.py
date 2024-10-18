import tkinter as tk
from tkinter import filedialog, messagebox
from vectorialRepresentationsGenerator import create_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from testProcessor import normalization, vectorize
import pickle
import math 

def load_vectorizer(file_path):
    with open(file_path, 'rb') as file:
        vectorizer, X = pickle.load(file)
    return vectorizer, X

def cosine(x, y):
	val = sum(x[index] * y[index] for index in range(len(x)))
	sr_x = math.sqrt(sum(x_val**2 for x_val in x))
	sr_y = math.sqrt(sum(y_val**2 for y_val in y))
	res = val/(sr_x*sr_y)
	return res

def compute_cosine_similarities(test_vector, X):
    similarities = []
    test_vector_dense = test_vector.toarray()[0]
    for index, vector in enumerate(X):
        vector_dense = vector.toarray()[0]
        similarity = cosine(test_vector_dense, vector_dense)
        similarities.append((index, similarity))
    return similarities

def normalize_text(file_path):
    return [normalization(file_path)]

# Lista de archivos de prueba
test_files = [
    'pruebas/Apple_titulo-contenido.txt',
    'pruebas/CFE_título-contenido_nuevo.txt',
    'pruebas/Liguilla_nuevo.txt',
    'pruebas/Papa_título.txt',
    'pruebas/Trump_contenido.txt'
]

# Normalizar textos de prueba
normalized_texts = [normalize_text(file) for file in test_files]

# Configuraciones de vectorizadores
vectorizer_configs = [
    ('titles/unigram/frecuency_vectorizer.pkl', 'test/unigram/title_frecuency_vectorizer_test.pkl'),
    ('titles/unigram/binarized_vectorizer.pkl', 'test/unigram/title_binarized_vectorizer_test.pkl'),
    ('titles/unigram/tfidf_vectorizer.pkl', 'test/unigram/title_tfidf_vectorizer.pkl'),
    ('contents/unigram/frecuency_vectorizer.pkl', 'test/unigram/content_frecuency_vectorizer_test.pkl'),
    ('contents/unigram/binarized_vectorizer.pkl', 'test/unigram/content_binarized_vectorizer_test.pkl'),
    ('contents/unigram/tfidf_vectorizer.pkl', 'test/unigram/content_tfidf_vectorizer.pkl'),
    ('both/unigram/frecuency_vectorizer.pkl', 'test/unigram/both_frecuency_vectorizer_test.pkl'),
    ('both/unigram/binarized_vectorizer.pkl', 'test/unigram/both_binarized_vectorizer_test.pkl'),
    ('both/unigram/tfidf_vectorizer.pkl', 'test/unigram/both_tfidf_vectorizer.pkl'),
    ('titles/bigram/frecuency_vectorizer.pkl', 'test/bigram/title_frecuency_vectorizer_test.pkl'),
    ('titles/bigram/binarized_vectorizer.pkl', 'test/bigram/title_binarized_vectorizer_test.pkl'),
    ('titles/bigram/tfidf_vectorizer.pkl', 'test/bigram/title_tfidf_vectorizer.pkl'),
    ('contents/bigram/frecuency_vectorizer.pkl', 'test/bigram/content_frecuency_vectorizer_test.pkl'),
    ('contents/bigram/binarized_vectorizer.pkl', 'test/bigram/content_binarized_vectorizer_test.pkl'),
    ('contents/bigram/tfidf_vectorizer.pkl', 'test/bigram/content_tfidf_vectorizer.pkl'),
    ('both/bigram/frecuency_vectorizer.pkl', 'test/bigram/both_frecuency_vectorizer_test.pkl'),
    ('both/bigram/binarized_vectorizer.pkl', 'test/bigram/both_binarized_vectorizer_test.pkl'),
    ('both/bigram/tfidf_vectorizer.pkl', 'test/bigram/both_tfidf_vectorizer.pkl')
]

# Procesar cada combinación de texto normalizado y configuración de vectorizador
for text in normalized_texts:
    all_similarities = []
    for vectorizer_path, test_vectorizer_path in vectorizer_configs:
        vectorizer, X = load_vectorizer(vectorizer_path)
        vectorizer, test_X = vectorize(text, vectorizer, test_vectorizer_path)
    
        # Calcular similitudes coseno
        similarities = compute_cosine_similarities(test_X, X)
        # Agregar la ruta del vectorizer_path a cada tupla de similitudes
        similarities_with_path = [(index, similarity, vectorizer_path) for index, similarity in similarities]
        all_similarities.extend(similarities_with_path)

    # Ordenar todas las similitudes y obtener el top 10
    all_similarities.sort(key=lambda x: x[1], reverse=True)
    top_10_similarities = all_similarities[:10]

    print(f'---------------------------- Top 10 similitudes para el texto ----------------------------')
    for similarity in top_10_similarities:
        print(similarity)