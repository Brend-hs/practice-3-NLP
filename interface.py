import tkinter as tk
from tkinter import filedialog, messagebox
from vectorialRepresentationsGenerator import create_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from testProcessor import normalization, vectorize
import pickle
import math 

def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        file_label.config(text=file_path)

def load_vectorizer(file_path):
    with open(file_path, 'rb') as file:
        vectorizer, X = pickle.load(file)
    return vectorizer, X

def cosine(x, y):
	val = sum(x[index] * y[index] for index in range(len(x)))
	sr_x = math.sqrt(sum(x_val**2 for x_val in x))
	sr_y = math.sqrt(sum(y_val**2 for y_val in y))
	res = val/(sr_x*sr_y)
	return (res)

def compute_cosine_similarities(test_vector, X):
    similarities = []
    for index, vector in enumerate(X):
        similarity = cosine(test_vector, vector)
        similarities.append((index, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:10]

def process_test_text(file_path, feature, representation, element):
    create_directory('test/unigram')
    create_directory('test/bigram')

    text_test = normalization(file_path)
    text_test = [text_test]

    print(f'Texto de prueba: {text_test}')

    if element == 'titulo':
        if feature == 'unigram':
            if representation == 'frecuencia':
                vectorizer, X = load_vectorizer('titles/unigram/frecuency_vectorizer.pkl')
                vectorizer, test_X = vectorize(text_test, vectorizer, 'test/unigram/frecuency_vectorizer_test.pkl')

            elif representation == 'binario':
                vectorizer, X = load_vectorizer('titles/unigram/binarized_vectorizer.pkl')
                vectorizer, test_X = vectorize(text_test, vectorizer, 'test/unigram/binarized_vectorizer_test.pkl')
            else:
                vectorizer, X = load_vectorizer('titles/unigram/tfidf_vectorizer.pkl')
                vectorizer, test_X = vectorize(text_test, vectorizer, 'test/unigram/tfidf_vectorizer.pkl')
        else:
            if representation == 'frecuencia':
                vectorizer, X = load_vectorizer('titles/bigram/frecuency_vectorizer.pkl')
                vectorizer, test_X = vectorize(text_test, vectorizer, 'test/bigram/frecuency_vectorizer_test.pkl')
            elif representation == 'binario':
                vectorizer, X = load_vectorizer('titles/bigram/binarized_vectorizer.pkl')
                vectorizer, test_X = vectorize(text_test, vectorizer, 'test/bigram/binarized_vectorizer_test.pkl')
            else:
                vectorizer, X = load_vectorizer('titles/bigram/tfidf_vectorizer.pkl')
                vectorizer, test_X = vectorize(text_test, vectorizer, 'test/bigram/tfidf_vectorizer.pkl')
    elif element == 'contenido':
        if feature == 'unigram':
            if representation == 'frecuencia':
                vectorizer, X = load_vectorizer('contents/unigram/frecuency_vectorizer.pkl')
                vectorizer, test_X = vectorize(text_test, vectorizer, 'test/unigram/frecuency_vectorizer_test.pkl')
            elif representation == 'binario':
                vectorizer, X = load_vectorizer('contents/unigram/binarized_vectorizer.pkl')
                vectorizer, test_X = vectorize(text_test, vectorizer, 'test/unigram/binarized_vectorizer_test.pkl')
            else:
                vectorizer, X = load_vectorizer('contents/unigram/tfidf_vectorizer.pkl')
                vectorizer, test_X = vectorize(text_test, vectorizer, 'test/unigram/tfidf_vectorizer.pkl')
        else:
            if representation == 'frecuencia':
                vectorizer, X = load_vectorizer('contents/bigram/frecuency_vectorizer.pkl')
                vectorizer, test_X = vectorize(text_test, vectorizer, 'test/bigram/frecuency_vectorizer_test.pkl')
            elif representation == 'binario':
                vectorizer, X = load_vectorizer('contents/bigram/binarized_vectorizer.pkl')
                vectorizer, test_X = vectorize(text_test, vectorizer, 'test/bigram/binarized_vectorizer_test.pkl')
            else:
                vectorizer, X = load_vectorizer('contents/bigram/tfidf_vectorizer.pkl')
                vectorizer, test_X = vectorize(text_test, vectorizer, 'test/bigram/tfidf_vectorizer.pkl')
    else:
        if feature == 'unigram':
            if representation == 'frecuencia':
                vectorizer, X = load_vectorizer('both/unigram/frecuency_vectorizer.pkl')
                vectorizer, test_X = vectorize(text_test, vectorizer, 'test/unigram/frecuency_vectorizer_test.pkl')
            elif representation == 'binario':
                vectorizer, X = load_vectorizer('both/unigram/binarized_vectorizer.pkl')
                vectorizer, test_X = vectorize(text_test, vectorizer, 'test/unigram/binarized_vectorizer_test.pkl')
            else:
                vectorizer, X = load_vectorizer('both/unigram/tfidf_vectorizer.pkl')
                vectorizer, test_X = vectorize(text_test, vectorizer, 'test/unigram/tfidf_vectorizer.pkl')
        else:
            if representation == 'frecuencia':
                vectorizer, X = load_vectorizer('both/bigram/frecuency_vectorizer.pkl')
                vectorizer, test_X = vectorize(text_test, vectorizer, 'test/bigram/frecuency_vectorizer_test.pkl')
            elif representation == 'binario':
                vectorizer, X = load_vectorizer('both/bigram/binarized_vectorizer.pkl')
                vectorizer, test_X = vectorize(text_test, vectorizer, 'test/bigram/binarized_vectorizer_test.pkl')
            else:
                vectorizer, X = load_vectorizer('both/bigram/tfidf_vectorizer.pkl')
                vectorizer, test_X = vectorize(text_test, vectorizer, 'test/bigram/tfidf_vectorizer.pkl')

    print(f'Vocabulario: {vectorizer.get_feature_names_out()}')
    print(f'Vectorizador: {test_X.toarray()}')

    # Compute cosine similarities
    top_similarities = compute_cosine_similarities(test_X.toarray()[0], X.toarray())
    print("Top 10 similares:")
    for index, similarity in top_similarities:
        print(f'Índice: {index+1}, Similitud: {similarity}')

    return top_similarities


def save_selected_options():
    feature = feature_var.get()
    representation = representation_var.get()
    element = element_var.get()
    file_path = file_label.cget("text")
    
    if not file_path:
        messagebox.showerror("Error", "Por favor, seleccione un archivo.")
        return
    
    if not feature or not representation or not element:
        messagebox.showerror("Error", "Por favor, complete todas las selecciones.")
        return
    
    print(f"Archivo seleccionado: {file_path}")
    print(f"Feature: {feature}")
    print(f"Representación: {representation}")
    print(f"Elemento: {element}")
    messagebox.showinfo("Información", "Selecciones guardadas correctamente.")

    results = process_test_text(file_path, feature, representation, element)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Top 10 similares:\n")
    for index, similarity in results:
        result_text.insert(tk.END, f'Índice: {index+1}, Similitud: {similarity}\n')

# Crear la ventana principal
root = tk.Tk()
root.title("Interfaz de Selección")

# Crear el botón para seleccionar un archivo
file_button = tk.Button(root, text="Seleccionar Archivo", command=select_file, font=("Arial", 12))
file_button.pack(pady=10)

# Etiqueta para mostrar la ruta del archivo seleccionado
file_label = tk.Label(root, text="", font=("Arial", 12))
file_label.pack(pady=5)

# Crear opciones de radio para seleccionar entre unigramas y bigramas
feature_var = tk.StringVar(value="unigram")
tk.Label(root, text="Seleccione Feature:", font=("Arial", 12)).pack(pady=5)
tk.Radiobutton(root, text="Unigrama", variable=feature_var, value="unigram", font=("Arial", 12)).pack(anchor=tk.W)
tk.Radiobutton(root, text="Bigrama", variable=feature_var, value="bigram", font=("Arial", 12)).pack(anchor=tk.W)

# Crear opciones de radio para seleccionar la representación
representation_var = tk.StringVar(value="frecuencia")
tk.Label(root, text="Seleccione Representación:", font=("Arial", 12)).pack(pady=5)
tk.Radiobutton(root, text="Frecuencia", variable=representation_var, value="frecuencia", font=("Arial", 12)).pack(anchor=tk.W)
tk.Radiobutton(root, text="Binario", variable=representation_var, value="binario", font=("Arial", 12)).pack(anchor=tk.W)
tk.Radiobutton(root, text="TF-IDF", variable=representation_var, value="tfidf", font=("Arial", 12)).pack(anchor=tk.W)

# Crear opciones de radio para seleccionar el elemento de comparación
element_var = tk.StringVar(value="titulo")
tk.Label(root, text="Seleccione Elemento de Comparación:", font=("Arial", 12)).pack(pady=5)
tk.Radiobutton(root, text="Título", variable=element_var, value="titulo", font=("Arial", 12)).pack(anchor=tk.W)
tk.Radiobutton(root, text="Contenido", variable=element_var, value="contenido", font=("Arial", 12)).pack(anchor=tk.W)
tk.Radiobutton(root, text="Título + Contenido", variable=element_var, value="both", font=("Arial", 12)).pack(anchor=tk.W)

# Crear el botón para generar la comparacion
save_button = tk.Button(root, text="Guardar Selecciones", command=save_selected_options, font=("Arial", 12))
save_button.pack(pady=20)

# Crear el widget de texto para mostrar los resultados
result_text = tk.Text(root, height=10, width=50, font=("Arial", 12))
result_text.pack(pady=10)

# Ejecutar la aplicación
root.mainloop()