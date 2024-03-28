import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(train_dir, validation_dir, img_size=(224, 224), batch_size=32):
    """
    Create data generators for training and validation images.

    Args:
        train_dir (str): The directory containing training images.
        validation_dir (str): The directory containing validation images.
        img_size (tuple, optional): The size to which all images will be resized. Defaults to (224, 224).
        batch_size (int, optional): The number of samples per gradient update. Defaults to 32.

    Returns:
        tuple: A tuple containing the training and validation data generators.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator, validation_generator
