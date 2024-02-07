import os
import shutil

# Actualizamos pip
os.system("python -m pip install --upgrade pip")

# Instalamos los requerimientos
os.system("pip install -r requirements.txt")

# Convertimos los notebooks
os.system("python convert_notebooks.py")