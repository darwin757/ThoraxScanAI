import os
import shutil
from sklearn.model_selection import train_test_split

# Define the paths
dataset_path = 'Lung X-Ray Image\Lung X-Ray Image'
new_dataset_path = 'ThoraxScanData'

# Categories
categories = ['Lung_Opacity', 'Normal', 'Viral Pneumonia']

# Split ratios
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15

# Create the NNDataset directory structure
for category in categories:
    os.makedirs(os.path.join(new_dataset_path, 'train', category), exist_ok=True)
    os.makedirs(os.path.join(new_dataset_path, 'validation', category), exist_ok=True)
    os.makedirs(os.path.join(new_dataset_path, 'test', category), exist_ok=True)

# Split and copy the files
for category in categories:
    # List the files in the category directory
    files = os.listdir(os.path.join(dataset_path, category))
    files = [os.path.join(dataset_path, category, f) for f in files]
    
    # Split the files
    train_files, test_files = train_test_split(files, test_size=1 - train_ratio, random_state=42)
    validation_files, test_files = train_test_split(test_files, test_size=test_ratio/(test_ratio + validation_ratio), random_state=42)

    # Function to copy files
    def copy_files(files, dataset_type):
        for file in files:
            shutil.copy(file, os.path.join(new_dataset_path, dataset_type, category))

    # Copy the files to their new locations
    copy_files(train_files, 'train')
    copy_files(validation_files, 'validation')
    copy_files(test_files, 'test')

print('Dataset successfully split and copied to ThoraxScanData.')
