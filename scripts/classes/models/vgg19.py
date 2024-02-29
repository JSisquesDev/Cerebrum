from .. import model

import tensorflow.keras as kr

class VGG19(model.Model):
    def __init__(self, name) -> None:
        self.name = name
    
    def create_model(self, img_height, img_width, img_deep, num_categories, activation='softmax'):
        self.model  = kr.Sequential([
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
            kr.layers.Dense(num_categories, activation=activation)
        ])