
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

def build_model(num_classes):
    """
    Builds a CNN model using VGG16 as the base for transfer learning, with integrated data augmentation.

    Parameters:
    - num_classes (int): Number of classes for the output layer.

    Returns:
    - TensorFlow model object ready for training.
    """
    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    base_model = VGG16(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    inputs = layers.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model
