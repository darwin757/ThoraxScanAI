# utils.py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_training_history(history):
    """
    Plots the training and validation accuracy and loss.

    Parameters:
    - history: A dictionary containing the training history. It should have the following keys:
        - 'accuracy': A list of training accuracy values for each epoch.
        - 'val_accuracy': A list of validation accuracy values for each epoch.
        - 'loss': A list of training loss values for each epoch.
        - 'val_loss': A list of validation loss values for each epoch.

    Returns:
    - None

    This function plots the training and validation accuracy and loss over the epochs. It creates a figure with two subplots. The first subplot shows the training and validation accuracy, while the second subplot shows the training and validation loss. The x-axis represents the epochs, and the y-axis represents the accuracy or loss values. The training accuracy and loss are plotted with blue circles connected by lines, while the validation accuracy and loss are plotted with red circles connected by lines. The legend is displayed to differentiate between the training and validation data.

    Example usage:
    history = {
        'accuracy': [0.5, 0.6, 0.7, 0.8],
        'val_accuracy': [0.4, 0.5, 0.6, 0.7],
        'loss': [1.0, 0.9, 0.8, 0.7],
        'val_loss': [1.2, 1.1, 1.0, 0.9]
    }
    plot_training_history(history)
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Acc')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

def display_images_with_predictions(images, labels, predictions=None, class_names=None):
    """
    Display images with their corresponding labels and predictions.

    Parameters:
    - images: A numpy array of shape (n_samples, height, width, channels) containing the images.
    - labels: A numpy array of shape (n_samples,) containing the labels for each image.
    - predictions: A numpy array of shape (n_samples,) containing the predictions for each image (default is None).
    - class_names: A list of length equal to the number of classes. Each element is a string representing the name of the class (default is None).

    Returns:
    - None

    This function displays a plot with 10 images from the input along with their labels and predictions. If the predictions array is provided, the function will also display the predictions. If the class_names array is provided, the function will use it to display the class names instead of default 'Class 0', 'Class 1', etc.
    """
    if not class_names:
        class_names = ['Class '+str(i) for i in range(len(np.unique(labels)))]
    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        if predictions is not None:
            plt.xlabel(f"True: {class_names[labels[i]]}\nPred: {class_names[predictions[i]]}")
        else:
            plt.xlabel(f"Label: {class_names[labels[i]]}")
    plt.show()

def load_and_preprocess_image(path, target_size=(224, 224)):
    """
    Load and preprocess an image.

    Parameters:
    - path: A string representing the path to the image file.
    - target_size: A tuple of two integers representing the desired target size (height, width) of the image (default is (224, 224)).

    Returns:
    - input_arr: A numpy array of shape (1, height, width, channels) containing the preprocessed image.

    This function loads an image using `tf.keras.preprocessing.image.load_img` and resizes it to the specified target size using `tf.keras.preprocessing.image.img_to_array`. It then converts the image to a batch by reshaping it to (1, height, width, channels) and applies preprocessing using `tf.keras.applications.vgg16.preprocess_input`.
    """
    image = tf.keras.preprocessing.image.load_img(path, target_size=target_size)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    input_arr = tf.keras.applications.vgg16.preprocess_input(input_arr)
    return input_arr
