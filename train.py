import tensorflow as tf
from model import build_model
from custom_data_loader import CustomDataLoader 
from utils import plot_training_history
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def main():
    """
    Runs the main training loop for the ThoraxScanAI model.

    Parameters:
    - None

    Returns:
    - None
    """
    # Setup paths
    dataset_path = 'ThoraxScanData'
    

     # Initialize the custom data loader
    data_loader = CustomDataLoader(dataset_path=dataset_path)
    train_ds, val_ds, test_ds = data_loader.get_data_loaders()  # Get TensorFlow datasets
    
    num_classes = 3  # Normal, Lung_Opacity, Viral Pneumonia
    model = build_model(num_classes)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    callbacks = [
        ModelCheckpoint('best_model.h5', save_best_only=True),
        EarlyStopping(patience=20),
        ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-3)
    ]

    # Train the model
    history = model.fit(
        train_ds,
        epochs=50,
        validation_data=val_ds,
        callbacks=callbacks)


    # Plot the training history
    plot_training_history(history)


if __name__ == '__main__':
        main()
