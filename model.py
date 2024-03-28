
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

def build_model(num_classes):
    """
    Builds a CNN model using VGG16 as the base for transfer learning.

    Parameters:
    - num_classes (int): Number of classes for the output layer.

    Returns:
    - TensorFlow model object ready for training.
    """
    base_model = VGG16(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model
