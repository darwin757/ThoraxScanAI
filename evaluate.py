import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import itertools
import matplotlib.pyplot as plt

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def create_test_generator(test_dir, img_size=(224, 224), batch_size=32):
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

def evaluate_model(model, test_generator):
    """
    Predicts on the test data using the given model and evaluates the performance using confusion matrix, classification report, and various metrics.
    
    Parameters:
    - model: The trained model to use for prediction.
    - test_generator: The generator for the test data.
    
    Returns:
    None
    """
    # Predict on the test data
    Y_pred = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(Y_pred, axis=1)

    # True labels
    y_true = test_generator.classes

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print('Confusion Matrix')
    print(cm)

    # Classification Report
    target_names = list(test_generator.class_indices.keys())
    print('Classification Report')
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Accuracy, Precision, Recall, F1-Score
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(f'Accuracy: {accuracy*100:.2f}%')
    print(f'Precision: {precision*100:.2f}%')
    print(f'Recall: {recall*100:.2f}%')
    print(f'F1-Score: {f1_score*100:.2f}%')

    # Plot Confusion Matrix
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(cm, classes=target_names, title='Confusion Matrix')

if __name__ == '__main__':
    model_path = 'best_model.h5'
    test_dir = 'ThoraxScanData/test'
    
    model = load_model(model_path)
    test_generator = create_test_generator(test_dir)
    evaluate_model(model, test_generator)
    plt.show()
