import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_batch(images: torch.Tensor, labels: torch.Tensor, idx_to_label: dict, num_images: int = 16) -> None:
    """
    Visualizes a batch of images with their labels.

    Args:
        images (torch.Tensor): Batch of images.
        labels (torch.Tensor): Batch of labels.
        idx_to_label (dict): Dictionary mapping label indices to label names.
        num_images (int): Number of images to visualize.
    """
    num_cols = 4
    num_rows = num_images // num_cols

    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        ax = plt.subplot(num_rows, num_cols, i + 1)
        inp = images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        ax.set_title(f'{idx_to_label[labels[i].item()]}', fontsize=8)
        ax.axis('off')

    plt.show()