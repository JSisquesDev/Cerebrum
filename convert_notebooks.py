
from dotenv import load_dotenv

import os
import shutil

if __name__ == '__main__':
    #Cargamos las variables de entorno
    load_dotenv()
    
    # Obtenemos la ruta donde estan los notebooks
    NOTEBOOKS_PATH = os.getenv("NOTEBOOKS_PATH")
    CODE_FILE_PATH = os.getenv("CODE_FILE_PATH")

    # Comprobamos si la ruta de los códigos existe
    if not os.path.exists(CODE_FILE_PATH):
        # Creamos la ruta de los códigos
        os.mkdir(CODE_FILE_PATH)

    # Convertimos los cuadernos .ipynb a archivos .py
    for notebook in os.listdir(NOTEBOOKS_PATH):
        try:
            print(f'Conviertiendo el archivo {notebook}')
            
            nb_name = notebook.split(".")[0]
            nb_path = os.path.join(NOTEBOOKS_PATH, notebook)
            
            
            os.system(f'jupyter nbconvert {nb_path} --to python')
            
            old_python_file = os.path.join(NOTEBOOKS_PATH, f'{nb_name}.py')
            new_python_file = os.path.join(CODE_FILE_PATH, f'{nb_name}.py')
            
            shutil.move(old_python_file, new_python_file)
            
            print(f'Archivo convertido con éxito')
        except:
            print(f'Error al procesar el archivo {notebook}. Puede que el archivo contenga un error o se encuentre vacío')
        