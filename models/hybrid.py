import torch
from torch import nn
from models.vit_optimized import ViTOptimized, EncoderLayer
from models.models import ResNetV2

class HybridModel(nn.Module):
    """
    Hybrid Model combining ResNet and Vision Transformer (ViT).

    This model uses a ResNet to extract feature maps, which are then processed by a Vision Transformer.

    Args:
        resnet (nn.Module): Pretrained ResNet model to extract features.
        vit (nn.Module): Vision Transformer model to process the extracted features.

    Returns:
        torch.Tensor: Output tensor with class scores.
    """
    def __init__(self, resnet: nn.Module, vit: nn.Module) -> None:
        super(HybridModel, self).__init__()
        self.resnet = resnet
        self.vit = vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Hybrid Model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        print(f"HybridModel input shape: {x.shape}")
        x = self.resnet(x)  # Extract feature maps using ResNet
        print(f"After resnet shape: {x.shape}")
        x = x.flatten(2).transpose(1, 2)  # Flatten and transpose for ViT input
        print(f"After flatten and transpose shape: {x.shape}")
        x = self.vit(x)  # Process feature maps using ViT
        return x

class ExtendedResNet(ResNetV2):
    """
    Extended ResNetV2 with an additional block for the hybrid model.

    Args:
        block_units (list): List of integers representing the number of blocks per unit.
        width_factor (int): Width factor for the model.
        out_channels (int): Number of output channels.

    Returns:
        torch.Tensor: Output tensor with reduced channels.
    """
    def __init__(self, block_units: list, width_factor: int, out_channels: int):
        super().__init__(block_units, width_factor)
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
        x = self.root(x)
        x = self.body.block1(x)
        x = self.body.block2(x)
        x = self.body.block3(x)  # Extract features up to the third block
        print(f"Before channel reducer shape: {x.shape}")
        x = self.channel_reducer(x)
        print(f"After channel reducer shape: {x.shape}")
        return x
