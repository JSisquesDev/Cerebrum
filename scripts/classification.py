import os
import tensorflow as tf
import matplotlib as plt
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from dotenv import find_dotenv, load_dotenv

from classes.dataset import Dataset
from classes.models.alexnet import AlexNet
from classes.models.vgg19 import VGG19

def format_path(path) -> str:
    return path.replace("/", os.sep).replace("\\", os.sep)

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
    
    dataset = Dataset(DATASET_PATH)
    dataset.configure(IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, "grayscale", 'categorical')
    dataset.apply_data_augmentation()
    dataset.set_data()
    
    # Mostramos las categorias
    print(f"Las categorias son: {dataset.labels}")
    print(f"Numero de categorias: {dataset.num_categories}")
    
    # Creamos las constantes para guardar el modelo
    MODEL_PATH = os.path.join(PROJECT_PATH, os.getenv("CLASSIFICATION_MODEL_PATH"))
    MODEL_NAME = os.path.join(MODEL_PATH, "AlexNet", "AlexNet")
    
    # Establecemos los epcohs y el patient
    EPOCHS = int(os.getenv("CLASSIFICATION_EPOCHS"))
    PATIENT = int(os.getenv("CLASSIFICATION_PATIENT"))
    
    alexnet = AlexNet(MODEL_NAME)
    
    alexnet.create_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEEP, dataset.num_categories)
    alexnet.compile(tf.optimizers.Adam(1e4), tf.metrics.Accuracy(), 'categorical_crossentropy')
    alexnet.set_early_stopping(PATIENT)
    alexnet.set_checkpoint()
    
    alexnet.train(dataset.train_data, EPOCHS, dataset.validation_data)
    
    # Evaluamos el modelo
    train_loss, train_success = alexnet.evaluate(dataset.train_data)
    validation_loss, validation_success = alexnet.evaluate(dataset.validation_data)
    
    alexnet.save(MODEL_NAME)
    
    MODEL_NAME = os.path.join(MODEL_PATH, "VGG19", "VGG19")
    
    vgg19 = VGG19(MODEL_NAME)
    
    vgg19.create_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEEP, dataset.num_categories)
    vgg19.compile(tf.optimizers.Adam(1e4), tf.metrics.Accuracy(), 'categorical_crossentropy')
    vgg19.set_early_stopping(PATIENT)
    vgg19.set_checkpoint()
    
    vgg19.train(dataset.train_data, EPOCHS, dataset.validation_data)
    
    # Evaluamos el modelo
    train_loss, train_success = vgg19.evaluate(dataset.train_data)
    validation_loss, validation_success = vgg19.evaluate(dataset.validation_data)
    
    vgg19.save(MODEL_NAME)