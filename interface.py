import tkinter as tk
from tkinter import filedialog, messagebox
from document_similarity import create_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from normalizationTest import normalization, vectorize

prueba = ''

def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        file_label.config(text=file_path)

def generate():
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
    
    # Aquí puedes guardar las selecciones en variables o hacer algo con ellas
    print(f"Archivo seleccionado: {file_path}")
    print(f"Feature: {feature}")
    print(f"Representación: {representation}")
    print(f"Elemento: {element}")
    messagebox.showinfo("Información", "Selecciones guardadas correctamente.")

    create_directory('test/unigram')
    create_directory('test/bigram')

    text_test = normalization(file_path)
    text_test = [text_test]

    if (feature == 'unigram'):
        if (representation == 'frecuencia'):
            if (element == 'titulo'):
                frecuency_vectorizer = CountVectorizer(token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(1, 1))
                frecuency_vectorizer, X_frecuency = vectorize(text_test, frecuency_vectorizer, 'test/unigram/frecuency_vectorizer_test.pkl')
            elif (element == 'contenido'):
                frecuency_vectorizer = CountVectorizer(token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(1, 1))
                frecuency_vectorizer, X_frecuency = vectorize(text_test, frecuency_vectorizer, 'test/unigram/frecuency_vectorizer_test.pkl')
            else:
                frecuency_vectorizer = CountVectorizer(token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(1, 1))
                frecuency_vectorizer, X_frecuency = vectorize(text_test, frecuency_vectorizer, 'test/unigram/frecuency_vectorizer_test.pkl')
        elif (representation == 'binario'):
            if (element == 'titulo'):
                frecuency_vectorizer = CountVectorizer(binary=True, token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(1, 1))
                frecuency_vectorizer, X_frecuency = vectorize(text_test, frecuency_vectorizer, 'test/unigram/binarized_vectorizer_test.pkl')
            elif (element == 'contenido'):
                frecuency_vectorizer = CountVectorizer(binary=True, token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(1, 1))
                frecuency_vectorizer, X_frecuency = vectorize(text_test, frecuency_vectorizer, 'test/unigram/binarized_vectorizer_test.pkl')
            else:
                frecuency_vectorizer = CountVectorizer(binary=True, token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(1, 1))
                frecuency_vectorizer, X_frecuency = vectorize(text_test, frecuency_vectorizer, 'test/unigram/binarized_vectorizer_test.pkl')
        else:
            if (element == 'titulo'):
                tfidf_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(1, 1))
                tfidf_vectorizer, X_tfidf = vectorize(text_test, tfidf_vectorizer, 'test/unigram/tfidf_vectorizer.pkl')
            elif (element == 'contenido'):
                tfidf_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(1, 1))
                tfidf_vectorizer, X_tfidf = vectorize(text_test, tfidf_vectorizer, 'test/unigram/tfidf_vectorizer.pkl')
            else:
                tfidf_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(1, 1))
                tfidf_vectorizer, X_tfidf = vectorize(text_test, tfidf_vectorizer, 'test/unigram/tfidf_vectorizer.pkl')
    else: 
        pass

# Crear la ventana principal
root = tk.Tk()
root.title("Interfaz de Selección")

# Crear el botón para seleccionar un archivo
file_button = tk.Button(root, text="Seleccionar Archivo", command=select_file)
file_button.pack(pady=10)

# Etiqueta para mostrar la ruta del archivo seleccionado
file_label = tk.Label(root, text="")
file_label.pack(pady=5)

# Crear opciones de radio para seleccionar entre unigramas y bigramas
feature_var = tk.StringVar(value="unigram")
tk.Label(root, text="Seleccione Feature:").pack(pady=5)
tk.Radiobutton(root, text="Unigrama", variable=feature_var, value="unigram").pack(anchor=tk.W)
tk.Radiobutton(root, text="Bigrama", variable=feature_var, value="bigram").pack(anchor=tk.W)

# Crear opciones de radio para seleccionar la representación
representation_var = tk.StringVar(value="frecuencia")
tk.Label(root, text="Seleccione Representación:").pack(pady=5)
tk.Radiobutton(root, text="Frecuencia", variable=representation_var, value="frecuencia").pack(anchor=tk.W)
tk.Radiobutton(root, text="Binario", variable=representation_var, value="binario").pack(anchor=tk.W)
tk.Radiobutton(root, text="TF-IDF", variable=representation_var, value="tfidf").pack(anchor=tk.W)

# Crear opciones de radio para seleccionar el elemento de comparación
element_var = tk.StringVar(value="titulo")
tk.Label(root, text="Seleccione Elemento de Comparación:").pack(pady=5)
tk.Radiobutton(root, text="Título", variable=element_var, value="titulo").pack(anchor=tk.W)
tk.Radiobutton(root, text="Contenido", variable=element_var, value="contenido").pack(anchor=tk.W)
tk.Radiobutton(root, text="Título + Contenido", variable=element_var, value="both").pack(anchor=tk.W)

# Crear el botón para generar la comparacion
save_button = tk.Button(root, text="Guardar Selecciones", command=generate)
save_button.pack(pady=20)

# Ejecutar la aplicación
root.mainloop()

print('prueba'+ prueba)
