import spacy
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Cargar el modelo de idioma en español de spacy
nlp = spacy.load('es_core_news_sm')

def normalization(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        text = file.readline()
    
    # Procesar el texto con SpaCy
    text_doc = nlp(text)

    # Remover las stop words y lematizar
    stop_pos = ['DET', 'ADP', 'CCONJ', 'SCONJ', 'PRON']
    text_tokens = [token.lemma_ for token in text_doc if token.pos_ not in stop_pos]
    
    return ' '.join(text_tokens)

# Vectorize
def vectorize(text, vectorizer, filename):
    if os.path.exists(filename):
        print(f'El vectorizador {filename} ya existe.')
        vector_file = open(filename, 'rb')
        vectorizer, X = pickle.load(vector_file)
        vector_file.close()
    else:
        vector_file = open(filename, 'wb')
        X = vectorizer.fit_transform(text)
        pickle.dump((vectorizer, X), vector_file)
        vector_file.close()
    return vectorizer, X