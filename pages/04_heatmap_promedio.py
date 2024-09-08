import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.datasets import mnist

# Cargar los datos de MNIST
(X_train, y_train), (_, _) = mnist.load_data()

# Calcular el promedio de las imágenes para cada dígito
mean_images = [np.mean(X_train[y_train == i], axis=0) for i in range(10)]

# Crear una figura para mostrar los mapas de calor
fig, ax = plt.subplots(2, 5, figsize=(10, 5))

# Mostrar los mapas de calor para las imágenes promedio de cada dígito usando solo Matplotlib
for i in range(10):
    ax[i // 5, i % 5].imshow(mean_images[i], cmap='viridis')
    ax[i // 5, i % 5].set_title(f'Dígito {i}')
    ax[i // 5, i % 5].axis('off')

# Ajustar el espaciado
plt.tight_layout()

# Mostrar la figura en Streamlit
st.title('Mapas de calor de las imágenes promedio por dígito en MNIST')
st.pyplot(fig)
