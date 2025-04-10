import os
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import multiprocessing
from functools import partial

def save_image(example, split, classes, output_directory):
    """
    Save a single image to the appropriate directory
    
    Args:
        example (dict): Dataset example
        split (str): Dataset split ('train' or 'test')
        classes (list): List of class names
        output_directory (str): Base output directory
    
    Returns:
        str: Path of saved image
    """
    # Get image and label
    img = example['img']
    label = example['label']
    class_name = classes[label]
    
    # Determine full output directory
    class_output_dir = os.path.join(output_directory, class_name)
    
    # Create class directory if it doesn't exist
    os.makedirs(class_output_dir, exist_ok=True)
    
    # Generate unique filename
    img_path = os.path.join(class_output_dir, f'{class_name}_{split}_{example["idx"]}.png')
    
    # Save image
    img.save(img_path)
    
    return img_path

def download_and_prepare_cifar10(root_dir, num_workers=None):
    """
    Download CIFAR-10 dataset from Hugging Face and save in ImageFolder format
    
    Args:
        root_dir (str): Root directory to save the dataset
        num_workers (int, optional): Number of workers for multiprocessing. 
                                     Defaults to None (uses all available cores)
    """
    # Create train and test directories
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # CIFAR-10 class names
    classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    # Load the dataset
    dataset = load_dataset('uoft-cs/cifar10', num_proc=24)

    # Add index to dataset for unique filenames
    for split in ['train', 'test']:
        dataset[split] = dataset[split].add_column('idx', range(len(dataset[split])))

    # Function to save images for a specific split
    def process_split(split):
        # Determine output directory
        output_directory = train_dir if split == 'train' else test_dir
        
        # Prepare partial function for multiprocessing
        save_func = partial(
            save_image, 
            split=split, 
            classes=classes, 
            output_directory=output_directory
        )
        
        # Use multiprocessing to save images
        with multiprocessing.Pool(processes=num_workers) as pool:
            # Use tqdm for progress tracking
            list(tqdm(
                pool.imap(save_func, dataset[split]), 
                total=len(dataset[split]), 
                desc=f'Saving {split} images'
            ))

    # Save train and test images
    process_split('train')
    process_split('test')

    print(f"CIFAR-10 dataset downloaded and saved to {root_dir}")

# Example usage
if __name__ == "__main__":
    # Specify the root directory where you want to save the dataset
    root_dir = './cifar10_dataset'
    
    # Download and prepare the dataset
    # You can specify number of workers, or leave as None to use all cores
    download_and_prepare_cifar10(root_dir, num_workers=None)

    # Verify dataset structure
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    # Define a simple transform (optional)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the dataset using ImageFolder
    dataset_train = datasets.ImageFolder(
        os.path.join(root_dir, 'train'), 
        transform=transform
    )

    # Print dataset info
    print(f"Total training images: {len(dataset_train)}")
    print(f"Classes: {dataset_train.classes}")