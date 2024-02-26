from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Dataset():
    def __init__(self, data_path) -> None:
        self.data_path = data_path
        self.image_data_generator = None
        self.train_data = None
        self.validation_data = None
        self.target_size = None,
        self.batch_size = None,
        self.color_mode = None,
        self.class_mode = None,
        self.labels = None
        self.num_categories = None
        
    def apply_data_augmentation(self, rescale=1.0/255, horizontal_flip=True, vertical_flip=True, validation_split=0.2, rotation_range=20):
        self.image_data_generator = ImageDataGenerator(rescale=rescale, horizontal_flip=horizontal_flip, vertical_flip=vertical_flip, validation_split=validation_split, rotation_range=rotation_range)
        return self.image_data_generator
    
    def configure(self, img_height, img_width, batch_size, color_mode, class_mode) -> None:
        self.target_size = (img_height, img_width)
        self.batch_size = batch_size
        self.color_mode = color_mode
        self.class_mode = class_mode
    
    def set_data(self):
        self.train_data = self.image_data_generator.flow_from_directory(
            self.data_path,
            target_size = self.target_size,
            batch_size = self.batch_size,
            color_mode = self.color_mode,
            shuffle = True,
            class_mode = self.class_mode,
            subset = 'training',
        )
        
        self.validation_data = self.image_data_generator.flow_from_directory(
            self.data_path,
            target_size = self.target_size,
            batch_size = self.batch_size,
            color_mode = self.color_mode,
            shuffle = False,
            class_mode = self.class_mode,
            subset = 'validation',
        )
        
        self.labels = self.train_data.class_indices
        self.num_categories = self.labels.__len__()
    
        return (self.train_data, self.validation_data)