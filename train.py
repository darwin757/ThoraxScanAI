import tensorflow as tf
from data_preprocessing import create_data_generators
from model import build_model
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
    train_dir = 'ThoraxScanData/train'
    validation_dir = 'ThoraxScanData/validation'
    num_classes = 3  # Normal, Lung_Opacity, Viral Pneumonia

    train_generator, validation_generator = create_data_generators(train_dir, validation_dir)
    
    model = build_model(num_classes)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
    early_stopping = EarlyStopping(patience=10)
    reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6)

    # Train the model
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stopping, reduce_lr])

if __name__ == '__main__':
    main()
