import os
import tensorflow as tf
import matplotlib as plt
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow.keras as kr
import random
import smtplib
import ssl
import time

from dotenv import find_dotenv, load_dotenv
from tensorflow.keras.utils import load_img, img_to_array
from numpy.core.defchararray import array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, model_from_json

from classes import vgg19
from classes import alexnet

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
    
    # Creamos las constantes para guardar el modelo
    MODEL_PATH = os.path.join(PROJECT_PATH, os.getenv("CLASSIFICATION_MODEL_PATH"))
    MODEL_NAME = os.path.join(MODEL_PATH, "AlexNet")
    
    # Establecemos los epcohs y el patient
    EPOCHS = int(os.getenv("CLASSIFICATION_EPOCHS"))
    PATIENT = int(os.getenv("CLASSIFICATION_PATIENT"))
    
    alexnet = alexnet.AlexNet(MODEL_NAME)
    
    alexnet.create_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEEP, NUM_CATEGORIES)
    alexnet.compile(tf.optimizers.Adam(1e4), tf.metrics.Accuracy(), 'categorical_crossentropy')
    alexnet.set_early_stopping(PATIENT)
    alexnet.set_checkpoint()
    
    alexnet.train(train_data, EPOCHS, validation_data)
    
    # Evaluamos el modelo
    train_loss, train_success = alexnet.evaluate(train_data)
    validation_loss, validation_success = alexnet.evaluate(validation_data)
    
    alexnet.save(MODEL_NAME)
    
    MODEL_NAME = os.path.join(MODEL_PATH, "VGG19")
    
    vgg19 = vgg19.VGG19(MODEL_NAME)
    
    vgg19.create_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEEP, NUM_CATEGORIES)
    vgg19.compile(tf.optimizers.Adam(1e4), tf.metrics.Accuracy(), 'categorical_crossentropy')
    vgg19.set_early_stopping(PATIENT)
    vgg19.set_checkpoint()
    
    vgg19.train(train_data, EPOCHS, validation_data)
    
    # Evaluamos el modelo
    train_loss, train_success = vgg19.evaluate(train_data)
    validation_loss, validation_success = vgg19.evaluate(validation_data)
    
    vgg19.save(MODEL_NAME)
    

