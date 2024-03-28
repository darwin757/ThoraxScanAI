# ThoraxScanAI

ThoraxScanAI is an advanced deep learning project aimed at classifying lung diseases from X-ray images using TensorFlow and Keras. The project utilizes pre-trained models for transfer learning, augmented with custom layers to tailor the network for lung disease detection.

## Project Structure

The project is structured into several key scripts, each fulfilling a specific role in the development and deployment of the machine learning model:

- `split_dataset.py`: Splits the original dataset into training, validation, and test sets.
- `data_preprocessing.py`: Prepares the data for training, including augmentation techniques.
- `model.py`: Defines the model architecture using VGG16 as a base for transfer learning.
- `train.py`: Orchestrates the model training process with callbacks for efficient learning.
- `evaluate.py`: Evaluates the trained model on a test set and generates performance metrics.
- `utils.py`: Contains utility functions such as plotting training history and processing single images for prediction.
- `predict_single_image.py`: Allows for classifying a single X-ray image using the trained model, demonstrating model usage on individual cases.

## Getting Started

## Getting Started

### Prerequisites

- Python 3.8 or later
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/darwin757/ThoraxScanAI.git
cd ThoraxScanAI
```

Install the required dependencies:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### Dataset

The original dataset, "Lung X-Ray Images", is processed and split into training, validation, and test sets using `split_dataset.py`. The script organizes the data into a structure suitable for training and evaluation.

### Training the Model

Run `train.py` to start the training process. The script will use the processed data and apply transfer learning with a VGG16 base model:

```bash
python train.py
```

### Evaluating the Model

After training, evaluate the model's performance on the test set with `evaluate.py`:

```bash
python evaluate.py
```

### Single Image Prediction

The project now includes a script for predicting the class of a single X-ray image using the trained model. After training, you can predict new images with:

```bash
python predict_single_image.py path/to/image.jpg
```

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Dataset sourced from [Kaggle - Lung X-Ray Images](https://www.kaggle.com/datasets/fatemehmehrparvar/lung-disease) by Md Alamin Talukder and Fatemeh Mehrparvar.
- ThoraxScanAI utilizes TensorFlow and the VGG16 model for its deep learning pipeline.

## Contact

For any queries or further discussion, please contact project maintainers at [ala.korabi@gmail.com].
