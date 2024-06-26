import tensorflow as tf
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

class CustomDataLoader:
    def __init__(self, dataset_path, img_size=(224, 224), batch_size=32, augment=False):
        """
        Initializes the object with the dataset path, image size, and batch size.

        Parameters:
            dataset_path (str): The path to the dataset.
            img_size (tuple): The size of the images (default is (224, 224)).
            batch_size (int): The batch size for training (default is 32).
        """
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.augment = augment
        self.class_names = self.get_class_names()

    def get_class_names(self):
        """
        Retrieve and return a sorted list of class names from the dataset path.
        Dynamically finds class names based on the directories within the 'train' subset.
        """
        train_path = os.path.join(self.dataset_path, 'train')  # Path to the training data
        class_names = sorted(name for name in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, name)))
        return class_names

    def parse_image(self, img_path):
        """
        Parses an image from the given image path.

        Parameters:
            img_path (str): The path to the image file.

        Returns:
            tf.Tensor: The parsed image tensor.
        """
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.img_size)
        return img

    def configure_for_performance(self, ds):
        """
        Configure the dataset for performance by applying caching, shuffling, batching, and prefetching operations.

        Args:
            self: The object instance.
            ds: The dataset to be configured for performance.

        Returns:
            The configured dataset.
        """
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def prepare_dataset(self, subset):
        """
        Prepare the dataset for training or validation.

        Parameters:
            subset (str): The subset of the dataset to prepare. It can be "train" or "val".

        Returns:
            tf.data.Dataset: The prepared dataset.
        """
        images, labels = [], []
        for class_name in self.class_names:
            class_dir = os.path.join(self.dataset_path, subset, class_name)
            class_label = self.class_names.index(class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                images.append(img_path)
                labels.append(class_label)
        
        path_ds = tf.data.Dataset.from_tensor_slices(images)
        image_ds = path_ds.map(self.parse_image, num_parallel_calls=AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
        
        # Apply one-hot encoding to the labels
        label_ds = label_ds.map(lambda label: tf.one_hot(label, depth=len(self.class_names)), num_parallel_calls=AUTOTUNE)
        
        ds = tf.data.Dataset.zip((image_ds, label_ds))
        
        if self.augment and subset == 'train':
            ds.map(self.augment_image, num_parallel_calls=AUTOTUNE)
        ds = self.configure_for_performance(ds)
        
        return ds

    def augment_image(self, image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1) # Adjust brightness by a small amount
        image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)) # Random rotation
        return image, label

    def get_data_loaders(self):
        """
        Returns three data loaders: train_ds, val_ds, and test_ds.
        """
        train_ds = self.prepare_dataset('train')
        val_ds = self.prepare_dataset('validation')
        test_ds = self.prepare_dataset('test')
        return train_ds, val_ds, test_ds

dataset_path = "ThoraxScanData"  
data_loader = CustomDataLoader(dataset_path=dataset_path, augment=True)
train_ds, val_ds, test_ds = data_loader.get_data_loaders()


