import numpy as np
from tensorflow.keras.models import load_model
from utils import load_and_preprocess_image
import matplotlib.pyplot as plt

# Path to the saved model and new image
model_path = 'best_model.h5'
image_path = 'ThoraxScanData/test/Viral Pneumonia/58.jpg'

# Load the trained model
model = load_model(model_path)

# Preprocess the image
preprocessed_image = load_and_preprocess_image(image_path)

# Predict the class of the image
prediction = model.predict(preprocessed_image)
predicted_class_index = np.argmax(prediction, axis=1)[0]

# Assuming you have a list of class names
class_names = ['Lung_Opacity', 'Normal', 'Viral Pneumonia']
predicted_class_name = class_names[predicted_class_index]

print(f"Predicted class: {predicted_class_name}")

# Optionally, if you want to display the image alongside the prediction
def display_image_with_prediction(image_path, predicted_class_name):
    """
    Display the image with its predicted class name.

    Parameters:
    - image_path: Path to the image file.
    - predicted_class_name: The name of the predicted class.
    """
    image = plt.imread(image_path)
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted: {predicted_class_name}")
    plt.axis('off')
    plt.show()

# Display the image and its prediction
display_image_with_prediction(image_path, predicted_class_name)
