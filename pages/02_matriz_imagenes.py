import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Cargar datos MNIST
(X_train, y_train), (_, _) = mnist.load_data()

# Función para mostrar la matriz 5x5 de imágenes aleatorias
def show_random_images():
    # Crear una figura para la matriz 5x5
    fig, axes = plt.subplots(4, 5, figsize=(5,5))
    
    # Muestreo aleatorio de imágenes
    random_indices = np.random.choice(len(X_train), 20, replace=False)
    
    # Iterar sobre la cuadrícula 5x5 y mostrar las imágenes aleatorias
    for i, ax in enumerate(axes.flat):
        # Seleccionar una imagen aleatoria
        random_image = X_train[random_indices[i]]
        random_label = random_indices[i]
        
        # Mostrar la imagen en la cuadrícula
        ax.imshow(random_image, cmap='gray')
        ax.set_title(f'index:{random_label}', fontsize=8)
        ax.axis('off')  # Ocultar los ejes
    
    # Ajustar el espaciado entre los subgráficos
    plt.tight_layout()
    st.pyplot(fig)

# Título de la aplicación
st.title('Matriz de 5x5 Imágenes de MNIST')

# Botón para actualizar la muestra aleatoria
if st.button('Generar nueva muestra aleatoria'):
    show_random_images()
else:
    show_random_images()
