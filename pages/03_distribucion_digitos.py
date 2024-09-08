import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

# Cargar los datos de MNIST
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# Dividir el conjunto de entrenamiento completo en conjunto de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Crear un DataFrame para almacenar las etiquetas y los conjuntos correspondientes
data = {
    'Conjunto': ['Entrenamiento'] * len(y_train) + ['Validación'] * len(y_val) + ['Prueba'] * len(y_test),
    'Dígito': np.concatenate([y_train, y_val, y_test])
}

df = pd.DataFrame(data)

# Crear la gráfica de barras utilizando Plotly
fig = px.histogram(df, x='Dígito', color='Conjunto', barmode='group',
                   title='Distribución de los dígitos en los conjuntos de entrenamiento, validación y prueba',
                   labels={'Dígito': 'Dígito', 'count': 'Frecuencia'},
                   category_orders={"Conjunto": ["Entrenamiento", "Validación", "Prueba"]})

# Mostrar la gráfica en Streamlit
st.title('Distribución de los dígitos en los conjuntos de entrenamiento, validación y prueba')
st.plotly_chart(fig, use_container_width=True)
