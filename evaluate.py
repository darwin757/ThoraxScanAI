import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import itertools
import matplotlib.pyplot as plt
from custom_data_loader import CustomDataLoader

model_path = 'best_model.h5'
dataset_path = 'ThoraxScanData'

def load_model(model_path):
    """
    Load a trained Keras model from the specified file path.

    Args:
        model_path (str): The path to the saved model file.

    Returns:
        tf.keras.Model: The loaded Keras model.

    """
    return tf.keras.models.load_model(model_path)

def create_test_generator(test_dir, img_size=(224, 224), batch_size=32):
    """
    Create a test data generator using the provided test directory, image size, and batch size.

    Parameters:
    - test_dir: A string representing the directory path where the test images are located.
    - img_size: A tuple representing the size of the images (default is (224, 224)).
    - batch_size: An integer representing the batch size for generating test data (default is 32).

    Returns:
    - test_generator: A data generator for test images.
    """
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    return test_generator

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def evaluate_model(model, test_ds, class_names):
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels, axis=1))

    # Generate and print the classification report
    print('Classification Report')
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print('Confusion Matrix')
    print(cm)

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(f'Accuracy: {accuracy*100:.2f}%')
    print(f'Precision: {precision*100:.2f}%')
    print(f'Recall: {recall*100:.2f}%')
    print(f'F1-Score: {f1_score*100:.2f}%')

    # Plot Confusion Matrix
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(cm, classes=data_loader.class_names, title='Confusion Matrix')

if __name__ == '__main__':
    model = load_model(model_path)
    data_loader = CustomDataLoader(dataset_path=dataset_path)
    _, _, test_ds = data_loader.get_data_loaders()
    class_names = data_loader.class_names
    evaluate_model(model, test_ds, class_names)
    plt.show()
