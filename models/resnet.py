import torch
from torch import nn
import requests
from pathlib import Path

# Download the ResNet model definition from the Google Research Big Transfer (BiT) repository
url = 'https://github.com/google-research/big_transfer/raw/master/bit_pytorch/models.py'
model_path = Path('models/models.py')
if not model_path.is_file():
    model_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url)
    with open(model_path, 'wb') as f:
        f.write(r.content)
    print('Model definition downloaded.')
else:
    print('Model definition already exists.')

# Import the ResNetV2 model definition
from models.models import ResNetV2  # This import statement assumes the model is downloaded to the `models` directory

class ExtendedResNet(nn.Module):
    """
    Extended ResNetV2 with an additional block for the hybrid model.

    This class extends the ResNetV2 model from the Google Research Big Transfer (BiT) repository to match the ViT input dimensions.

    Args:
        block_units (list): List of integers representing the number of blocks per unit.
        width_factor (int): Width factor for the model.
        out_channels (int): Number of output channels.
    """

    def __init__(self, block_units: list, width_factor: int, out_channels: int):
        super().__init__()
        self.resnet = ResNetV2(block_units, width_factor)  # Initialize the ResNetV2 model
        self.out_channels = out_channels

        # Channel reducer to match the ViT input
        self.channel_reducer = nn.Conv2d(1024, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Extended ResNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = self.resnet(x)
        x = self.channel_reducer(x)
        return x

# Note: The `ResNetV2` class is defined in the Google Research Big Transfer (BiT) repository and is assumed to be available after downloading.