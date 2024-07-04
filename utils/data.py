import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split, Subset
from collections import defaultdict

def get_transform(image_size: int) -> transforms.Compose:
    """
    Returns the transform to be applied to the dataset.

    Args:
        image_size (int): The size to which the image will be resized.

    Returns:
        transforms.Compose: Composed transformation.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_subset(dataset: CIFAR10, num_images_per_class: int) -> Subset:
    """
    Creates a subset of the dataset with a specified number of images per class.

    Args:
        dataset (CIFAR10): The dataset from which to create the subset.
        num_images_per_class (int): The number of images per class.

    Returns:
        Subset: A subset of the dataset with the specified number of images per class.
    """
    class_count = defaultdict(int)
    indices = []

    for idx, (_, label) in enumerate(dataset):
        if class_count[label] < num_images_per_class:
            class_count[label] += 1
            indices.append(idx)
        if len(indices) == num_images_per_class * len(class_count):
            break

    return Subset(dataset, indices)

def get_dataloaders(image_size: int, batch_size: int, num_images_per_class: int, train_ratio: float = 0.8) -> (DataLoader, DataLoader):
    """
    Returns the train and test dataloaders.

    Args:
        image_size (int): The size to which the image will be resized.
        batch_size (int): The batch size.
        num_images_per_class (int): The number of images per class.
        train_ratio (float): The ratio of the train dataset.

    Returns:
        tuple: Train and test dataloaders.
    """
    transform = get_transform(image_size)
    dataset = CIFAR10(root='datasets', download=True, transform=transform, train=True)
    
    subset = create_subset(dataset, num_images_per_class)
    train_size = int(train_ratio * len(subset))
    test_size = len(subset) - train_size

    train_dataset, test_dataset = random_split(subset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader, test_dataloader