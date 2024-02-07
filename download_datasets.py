from dotenv import load_dotenv
import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import shutil
import time

def download_dataset(dataset, path):
    # Aseguramos que la ruta exista
    os.makedirs(path, exist_ok=True)
       
    # Nos autenticamos con la API de Kaggle
    api = KaggleApi()
    api.authenticate()

    # Descargamos los datos
    print(f'Descargando dataset {dataset}...')
    os.system(f'kaggle datasets download {dataset} --force')
    
    # Obtenemos el autor y el nombre del dataset
    AUTHOR = dataset.split("/")[0]
    DATASET_NAME = dataset.split("/")[1]

    # Copiamos el ZIP descargado a la carpeta correspondiente
    shutil.move(f'{DATASET_NAME}.zip', os.path.join(path, f'{DATASET_NAME}.zip'))
    
    # Obtenemos el zip descargado
    ZIP_PATH = os.path.join(path, f'{DATASET_NAME}.zip')

    # Descomprimimos el ZIP
    print(f'Descomprimiendo el dataset {DATASET_NAME}...')
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip:
        zip.extractall(path)

    # Borramos el ZIP descargado
    os.remove(ZIP_PATH)
    print(f'El dataset {dataset} se ha descargado correctamente')
    
    # En el caso de estar descargando en dataset de detección
    if  dataset == os.getenv("DATASET_DETECTION"):
        # Borramos las dos carpetas padre
        shutil.move(os.getenv("OLD_BRAIN_TUMOR_PATH"), os.getenv("BRAIN_TUMOR_PATH"))
        shutil.move(os.getenv("OLD_BRAIN_HEALTH_PATH"), os.getenv("BRAIN_HEALTH_PATH"))
        shutil.rmtree(os.getenv("OLD_BRAIN_DATASET_PARENT_PATH"))
        
    # En el caso de estar descargando en dataset de segmentación
    if  dataset == os.getenv("DATASET_SEGMENTATION"):
        # Borramos las dos carpetas padre
        shutil.move(os.getenv("OLD_SEGMENTATION_PATH"), os.getenv("NEW_SEGMENTATION_PATH"))
        shutil.rmtree(os.getenv("OLD_SEGMENTATION_PARENT_PATH"))


def create_kaggle_file(username, key) -> None:
    # Obtenemos la ruta de kaggle
    KAGGLE_PATH = os.path.expanduser('~/.kaggle')
    
    # Creamos la carpeta kaggle si no existe
    os.makedirs(KAGGLE_PATH, exist_ok=True)

    # Obtenemos la ruta para el archivo JSON de Kaggle
    FILE_PATH = os.path.join(KAGGLE_PATH, 'kaggle.json')

    # Guardamos las credenciales en kaggle.json
    with open(FILE_PATH, 'w') as json_file:
        json.dump({"username": username, "key": key}, json_file)

    print(f'Creado el archivo de credenciales de Kaggle en {FILE_PATH}')

if __name__ == '__main__':
    start_time = time.time()
    
    # Cargamos las variables de entorno
    load_dotenv()
    
    USERNAME = os.getenv('KAGGLE_USERNAME')
    KEY = os.getenv('KAGGLE_API_KEY')
    
    create_kaggle_file(USERNAME, KEY)
    
    # Nombre de los datasets
    DATASET_DETECTION = os.getenv('DATASET_DETECTION')
    DATASET_CLASSIFICATION = os.getenv('DATASET_CLASSIFICATION')
    DATASET_SEGMENTATION = os.getenv('DATASET_SEGMENTATION')
   
    # Rutas de los dataset
    DATASET_DETECTION_PATH = os.getenv('DATASET_DETECTION_PATH')
    DATASET_CLASSIFICATION_PATH = os.getenv('DATASET_CLASSIFICATION_PATH')
    DATASET_SEGMENTATION_PATH = os.getenv('DATASET_SEGMENTATION_PATH')
    
    # Descargamos los datasets
    download_dataset(DATASET_DETECTION, DATASET_DETECTION_PATH)
    download_dataset(DATASET_CLASSIFICATION, DATASET_CLASSIFICATION_PATH)
    download_dataset(DATASET_SEGMENTATION, DATASET_SEGMENTATION_PATH)
    
    end_time = time.time() - start_time
    
    print(f'Todos los datasets se han descargado, tiempo total {end_time} segundos')
    
    

