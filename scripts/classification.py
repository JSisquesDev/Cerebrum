import os
import tensorflow as tf
import matplotlib as plt
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow.keras as kr
import random

from dotenv import find_dotenv, load_dotenv
from tensorflow.keras.utils import load_img, img_to_array
from numpy.core.defchararray import array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

def format_path(path) -> str:
    return path.replace("/", os.sep).replace("\\", os.sep)

def set_image_data_generator() -> ImageDataGenerator:
    return ImageDataGenerator(
        rescale=1.0/255,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,
        rotation_range=20 
    )

def create_model(img_height, img_width, img_deep, num_categories):
    # Configuramos el modelo
    optimizer = tf.optimizers.Adam(1e4)
    metrics = tf.metrics.Accuracy()
    loss = 'categorical_crossentropy'
    
    model  = kr.Sequential([
    # Bloque 01
    kr.layers.Conv2D(64, activation='relu', kernel_size=(3, 3), input_shape=(img_height, img_width, img_deep)),
    kr.layers.Conv2D(64, activation='relu', kernel_size=(3, 3), padding='same'),
    kr.layers.MaxPooling2D(2,2),

    # Bloque 02
    kr.layers.Conv2D(128, activation='relu', kernel_size=(3, 3), padding='same'),
    kr.layers.Conv2D(128, activation='relu', kernel_size=(3, 3), padding='same'),
    kr.layers.MaxPooling2D(2,2),

    # Bloque 03
    kr.layers.Conv2D(256, activation='relu', kernel_size=(3, 3), padding='same'),
    kr.layers.Conv2D(256, activation='relu', kernel_size=(3, 3), padding='same'),
    kr.layers.Conv2D(256, activation='relu', kernel_size=(3, 3), padding='same'),
    kr.layers.Conv2D(256, activation='relu', kernel_size=(3, 3), padding='same'),
    kr.layers.MaxPooling2D(2,2),
    
    # Bloque 04
    kr.layers.Conv2D(512, activation='relu', kernel_size=(3, 3), padding='same'),
    kr.layers.Conv2D(512, activation='relu', kernel_size=(3, 3), padding='same'),
    kr.layers.Conv2D(512, activation='relu', kernel_size=(3, 3), padding='same'),
    kr.layers.Conv2D(512, activation='relu', kernel_size=(3, 3), padding='same'),
    kr.layers.MaxPooling2D(2,2),
    
    # Bloque 05
    kr.layers.Conv2D(512, activation='relu', kernel_size=(3, 3), padding='same'),
    kr.layers.Conv2D(512, activation='relu', kernel_size=(3, 3), padding='same'),
    kr.layers.Conv2D(512, activation='relu', kernel_size=(3, 3), padding='same'),
    kr.layers.Conv2D(512, activation='relu', kernel_size=(3, 3), padding='same'),
    kr.layers.MaxPooling2D(2,2),

    # Bloque 06
    kr.layers.Flatten(),
    kr.layers.Dropout(0.5),

    # Bloque 07
    kr.layers.Dense(4096, activation='relu'),
    kr.layers.Dropout(0.5),
    kr.layers.Dense(4096, activation='relu'),
    kr.layers.Dropout(0.5),
    kr.layers.Dense(num_categories, activation='softmax')
    ])
    
    model.compile(
        optimizer = optimizer,
        metrics = metrics,
        loss = loss
    )
    
    print(f"Resumen del modelo: {model.summary()}")
    
    return model

if __name__ == '__main__':
    
    # Cargamos las variables de entorno
    load_dotenv(find_dotenv())
    
    # Comprobamos si se está usando la GPU
    print(f"Dispositivo de entrenamiento: {tf.config.list_physical_devices('GPU')}")
    
    # Establecemos el tamaño de las imágenes
    IMG_WIDTH = int(os.getenv("CLASSIFICATION_IMG_WIDTH"))
    IMG_HEIGHT = int(os.getenv("CLASSIFICATION_IMG_HEIGHT"))
    IMG_DEEP = int(os.getenv("CLASSIFICATION_IMG_DEEP"))
    BATCH_SIZE = int(os.getenv("CLASSIFICATION_BATCH_SIZE"))
    
    # Establecemos las constantes de las rutas
    PROJECT_PATH = os.getenv("PROJECT_PATH")
    DATASET_PATH = os.path.join(PROJECT_PATH, os.getenv("DATASET_CLASSIFICATION_PATH"))
    
    print(f"Ruta de entreno: {DATASET_PATH}")
    
    # Creamos el generador de datos
    data_generator = set_image_data_generator()
    
    # Creamos el dataset de entrenamiento
    train_data = data_generator.flow_from_directory(
        DATASET_PATH,
        target_size = (IMG_HEIGHT, IMG_WIDTH),
        batch_size = BATCH_SIZE,
        color_mode = "grayscale",
        shuffle = True,
        class_mode = 'categorical',
        subset = 'training',
    )
    
    # Creamos el dataset de validación
    validation_data = data_generator.flow_from_directory(
        DATASET_PATH,
        target_size = (IMG_HEIGHT, IMG_WIDTH),
        batch_size = BATCH_SIZE,
        color_mode = "grayscale",
        shuffle = False,
        class_mode = 'categorical',
        subset = 'validation'
    )
    
    # Mostramos las categorias
    LABELS = train_data.class_indices
    NUM_CATEGORIES = LABELS.__len__()
    print(f"Las categorias son: {LABELS}")
    print(f"Numero de categorias: {NUM_CATEGORIES}")
    
    # Creamos el modelo (VGG19)
    model = create_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEEP, NUM_CATEGORIES)

    # Establecemos los epcohs y el patient
    EPOCHS = int(os.getenv("CLASSIFICATION_EPOCHS"))
    PATIENT = int(os.getenv("CLASSIFICATION_PATIENT"))
    
    # Creamos las constantes para guardar el modelo
    MODEL_PATH = os.path.join(PROJECT_PATH, os.getenv("CLASSIFICATION_MODEL_PATH"))
    MODEL_NAME = os.path.join(MODEL_PATH, os.getenv("CLASSIFICATION_MODEL_NAME"))
    
    # Establecemos los checkpoints
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENT, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_NAME, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    
    # Entrenamos el modelo
    result = model.fit(train_data, epochs=EPOCHS, callbacks=[early_stopping, checkpoint], validation_data=validation_data)
    
    model.save(MODEL_NAME)
    
    # Obtenemos las estadisticas del modelo
    scores = model.evaluate(train_data, validation_data, verbose=0)
    accuracy = scores[1]*100
    
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))