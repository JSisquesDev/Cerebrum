import time

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class Model():
    def __init__(self, name) -> None:
        self.model = None
        self.early_stopping = None
        self.checkpoint = None
        self.result = None
        self.name = name
        self.train_time = None
    
    def compile(self, optimizer, metrics, loss):
        self.model.compile(
            optimizer = optimizer,
            metrics = metrics,
            loss = loss
        )
        
        print(f"Resumen del modelo: {self.model.summary()}")
        
    def set_early_stopping(self, patience, restore_best_weights=True) -> None:
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=restore_best_weights)
    
    def set_checkpoint(self) -> None:
        self.checkpoint = ModelCheckpoint(f"{self.name}.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    def train(self, train_data, epochs, validation_data):
        start = time.time()
        self.result = self.model.fit(train_data, epochs=epochs, callbacks=[self.early_stopping, self.checkpoint], validation_data=validation_data)
        end = time.time() - start

        self.convert_time(end)

        # Mostramos el tiempo total de entreno
        print(f"Tiempo total de entrenamiento: {self.train_time}")
        
        return self.result
    
    def convert_time(self, time):
        self.train_time = time / 60
    
    def evaluate(self, data):
        # Se puede comprobar con el train data, valid data y test data
        loss, success = self.model.evaluate(data)
        
        print(f'Perdida: {loss}')
        print(f'Acierto: {success}')
        
        return (loss, success)   
     
    def save(self, name):
        # Serializamos el modelo en JSON
        model_json = self.model.to_json()
        with open(f"{name}.json", "w") as json_file:
            json_file.write(model_json)
            
        # Serializamos el modelo en H5
        self.model.save_weights(f"{name}.h5")
        
        print(f"Modelo {name} guardado en el disco")