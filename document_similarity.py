import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os
import numpy as np

def read_csv_file():
    # Leer el archivo CSV y devolver los datos como una lista de listas
    with open('normalized_data_corpus.csv', mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        data = list(reader)
    return data

# Extraer los títulos y contenidos de los datos
def extract_titles_and_contents(data):
    titles = [row[1] for row in data]
    contents = [row[2] for row in data]
    both = [row[1] + ' ' + row[2] for row in data]
    return titles, contents, both

# Vectorizar los títulos y contenidos y guardar el vectorizador
def vectorize_and_save(titles, vectorizer, filename):
    corpus = titles

    if os.path.exists(filename):
        print(f'El vectorizador {filename} ya existe.')
        vector_file = open(filename, 'rb')
        vectorizer, X = pickle.load(vector_file)
        vector_file.close()
    else:
        vector_file = open(filename, 'wb')
        X = vectorizer.fit_transform(corpus)
        pickle.dump((vectorizer, X), vector_file)
        vector_file.close()
    return vectorizer, X

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_vectorial_representations():
    data = read_csv_file()
    titles, contents, both = extract_titles_and_contents(data)

    # Crear directorios para guardar los archivos
    create_directory('titles/unigram')
    create_directory('titles/bigram')
    create_directory('contents/unigram')
    create_directory('contents/bigram')
    create_directory('both/unigram')
    create_directory('both/bigram')

    # Title - Frecuency vectorial representation - Unigram
    frecuency_vectorizer = CountVectorizer(token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(1, 1))
    frecuency_vectorizer, X_frecuency = vectorize_and_save(titles, frecuency_vectorizer, 'titles/unigram/frecuency_vectorizer.pkl')

    # Title - Binarized vectorial representation - Unigram
    binarized_vectorizer = CountVectorizer(binary=True, token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(1, 1))
    binarized_vectorizer, X_binarized = vectorize_and_save(titles, binarized_vectorizer, 'titles/unigram/binarized_vectorizer.pkl')

    # Title - TF-IDF vectorial representation - Unigram
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(1, 1))
    tfidf_vectorizer, X_tfidf = vectorize_and_save(titles, tfidf_vectorizer, 'titles/unigram/tfidf_vectorizer.pkl')

    # Set the print options to print the entire array
    # np.set_printoptions(threshold=np.inf)

    print('--------------------------------------------TITLE UNIGRAM--------------------------------------------')
    # Print the vocabulary and vectorizer for each vectorial representation
    for name, vectorizer, X in [
        ('Frecuency', frecuency_vectorizer, X_frecuency),
        ('Binarized', binarized_vectorizer, X_binarized),
        ('TF-IDF', tfidf_vectorizer, X_tfidf)
    ]:
        print(f'Vocabulario {name}:', vectorizer.get_feature_names_out())
        print(f'Vectorizador {name}:', X.toarray())

    # Title - Frecuency vectorial representation - Bigram
    frecuency_vectorizer = CountVectorizer(token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(2, 2))
    frecuency_vectorizer, X_frecuency = vectorize_and_save(titles, frecuency_vectorizer, 'titles/bigram/frecuency_vectorizer.pkl')

    # Title - Binariazed vectorial representation - Bigram
    binarized_vectorizer = CountVectorizer(binary=True, token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(2, 2))
    binarized_vectorizer, X_binarized = vectorize_and_save(titles, binarized_vectorizer, 'titles/bigram/binarized_vectorizer.pkl')

    # Title - TF-IDF vectorial representation - Bigram
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(2, 2))
    tfidf_vectorizer, X_tfidf= vectorize_and_save(titles, tfidf_vectorizer, 'titles/bigram/tfidf_vectorizer.pkl')

    print('--------------------------------------------TITLE BIGRAM--------------------------------------------')
    # Print the vocabulary and vectorizer for each vectorial representation
    for name, vectorizer, X in [
        ('Frecuency', frecuency_vectorizer, X_frecuency),
        ('Binarized', binarized_vectorizer, X_binarized),
        ('TF-IDF', tfidf_vectorizer, X_tfidf)
    ]:
        print(f'Vocabulario {name}:', vectorizer.get_feature_names_out())
        print(f'Vectorizador {name}:', X.toarray())
    
    # Content - Frecuency vectorial representation - Unigram
    frecuency_vectorizer = CountVectorizer(token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(1, 1))
    frecuency_vectorizer, X_frecuency = vectorize_and_save(contents, frecuency_vectorizer, 'contents/unigram/frecuency_vectorizer.pkl')

    # Content - Binarized vectorial representation - Unigram
    binarized_vectorizer = CountVectorizer(binary=True, token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(1, 1))
    binarized_vectorizer, X_binarized = vectorize_and_save(contents, binarized_vectorizer, 'contents/unigram/binarized_vectorizer.pkl')

    # Content - TF-IDF vectorial representation - Unigram
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(1, 1))
    tfidf_vectorizer, X_tfidf = vectorize_and_save(contents, tfidf_vectorizer, 'contents/unigram/tfidf_vectorizer.pkl')

    print('--------------------------------------------CONTENT UNIGRAM--------------------------------------------')
    # Print the vocabulary and vectorizer for each vectorial representation
    for name, vectorizer, X in [
        ('Frecuency', frecuency_vectorizer, X_frecuency),
        ('Binarized', binarized_vectorizer, X_binarized),
        ('TF-IDF', tfidf_vectorizer, X_tfidf)
    ]:
        print(f'Vocabulario {name}:', vectorizer.get_feature_names_out())
        print(f'Vectorizador {name}:', X.toarray())

    # Content - Frecuency vectorial representation - Bigram
    frecuency_vectorizer = CountVectorizer(token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(2, 2))
    frecuency_vectorizer, X_frecuency = vectorize_and_save(contents, frecuency_vectorizer, 'contents/bigram/frecuency_vectorizer.pkl')

    # Content - Binariazed vectorial representation - Bigram
    binarized_vectorizer = CountVectorizer(binary=True, token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(2, 2))
    binarized_vectorizer, X_binarized = vectorize_and_save(contents, binarized_vectorizer, 'contents/bigram/binarized_vectorizer.pkl')

    # Content - TF-IDF vectorial representation - Bigram
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(2, 2))
    tfidf_vectorizer, X_tfidf= vectorize_and_save(contents, tfidf_vectorizer, 'contents/bigram/tfidf_vectorizer.pkl')

    print('--------------------------------------------CONTENT BIGRAM--------------------------------------------')
    # Print the vocabulary and vectorizer for each vectorial representation
    for name, vectorizer, X in [
        ('Frecuency', frecuency_vectorizer, X_frecuency),
        ('Binarized', binarized_vectorizer, X_binarized),
        ('TF-IDF', tfidf_vectorizer, X_tfidf)
    ]:
        print(f'Vocabulario {name}:', vectorizer.get_feature_names_out())
        print(f'Vectorizador {name}:', X.toarray())

    # Both - Frecuency vectorial representation - Unigram
    frecuency_vectorizer = CountVectorizer(token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(1, 1))
    frecuency_vectorizer, X_frecuency = vectorize_and_save(both, frecuency_vectorizer, 'both/unigram/frecuency_vectorizer.pkl')

    # Both - Binarized vectorial representation - Unigram
    binarized_vectorizer = CountVectorizer(binary=True, token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(1, 1))
    binarized_vectorizer, X_binarized = vectorize_and_save(both, binarized_vectorizer, 'both/unigram/binarized_vectorizer.pkl')

    # Both - TF-IDF vectorial representation - Unigram
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(1, 1))
    tfidf_vectorizer, X_tfidf = vectorize_and_save(both, tfidf_vectorizer, 'both/unigram/tfidf_vectorizer.pkl')

    print('--------------------------------------------BOTH UNIGRAM--------------------------------------------')
    # Print the vocabulary and vectorizer for each vectorial representation
    for name, vectorizer, X in [
        ('Frecuency', frecuency_vectorizer, X_frecuency),
        ('Binarized', binarized_vectorizer, X_binarized),
        ('TF-IDF', tfidf_vectorizer, X_tfidf)
    ]:
        print(f'Vocabulario {name}:', vectorizer.get_feature_names_out())
        print(f'Vectorizador {name}:', X.toarray())

    # Both - Frecuency vectorial representation - Bigram
    frecuency_vectorizer = CountVectorizer(token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(2, 2))
    frecuency_vectorizer, X_frecuency = vectorize_and_save(both, frecuency_vectorizer, 'both/bigram/frecuency_vectorizer.pkl')

    # Both - Binariazed vectorial representation - Bigram
    binarized_vectorizer = CountVectorizer(binary=True, token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(2, 2))
    binarized_vectorizer, X_binarized = vectorize_and_save(both, binarized_vectorizer, 'both/bigram/binarized_vectorizer.pkl')

    # Both - TF-IDF vectorial representation - Bigram
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(2, 2))
    tfidf_vectorizer, X_tfidf= vectorize_and_save(both, tfidf_vectorizer, 'both/bigram/tfidf_vectorizer.pkl')

    print('--------------------------------------------BOTH BIGRAM--------------------------------------------')
    # Print the vocabulary and vectorizer for each vectorial representation
    for name, vectorizer, X in [
        ('Frecuency', frecuency_vectorizer, X_frecuency),
        ('Binarized', binarized_vectorizer, X_binarized),
        ('TF-IDF', tfidf_vectorizer, X_tfidf)
    ]:
        print(f'Vocabulario {name}:', vectorizer.get_feature_names_out())
        print(f'Vectorizador {name}:', X.toarray())

