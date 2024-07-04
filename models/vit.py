import torch
from torch import nn
from einops import rearrange, repeat

class ViTEmbedding(nn.Module):
    """
    Vision Transformer Embedding Layer

    This class defines the embedding layer for the Vision Transformer (ViT). It splits the input image into patches, projects them into a latent space, and adds positional embeddings.

    Args:
        image_size (int): The height and width of the input image.
        patch_size (int): The size of each image patch.
        channels (int): The number of channels in the input image.
        embed_dim (int): The dimensionality of the latent space.
        device (torch.device): The device to run the computations on.
    """

    def __init__(self, image_size, patch_size, channels, embed_dim, device):
        super(ViTEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.embed_dim = embed_dim
        self.device = device

        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels=channels,
                              out_channels=embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size,
                              bias=True)

        self.class_token = nn.Parameter(torch.randn(
            size=(1, 1, embed_dim),
            requires_grad=True,
            device=device
        ))

        self.positional_embedding = nn.Parameter(torch.randn(
            size=(1, self.num_patches + 1, embed_dim),
            requires_grad=True,
            device=device
        ))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the embedding layer.

        Args:
            image (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Embedded image tensor of shape (batch_size, num_patches + 1, embed_dim).
        """
        batch_size = image.shape[0]
        image = self.proj(image.to(self.device))
        image = image.flatten(2)
        image = image.permute(0, 2, 1)

        class_token = self.class_token.expand(batch_size, -1, -1)
        image = torch.cat((class_token, image), dim=1)
        image = image + self.positional_embedding

        return image

class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer

    This class defines a single encoder layer for the Vision Transformer. It includes multi-head self-attention and a feed-forward neural network.

    Args:
        embed_dim (int): The dimensionality of the input and output embeddings.
        num_heads (int): The number of attention heads.
        mlp_dim (int): The dimensionality of the feed-forward network.
        dropout (float): The dropout rate.
        device (torch.device): The device to run the computations on.
    """

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1, device=None):
        super(TransformerEncoderLayer, self).__init__()
        self.device = device

        self.norm1 = nn.LayerNorm(embed_dim)
        self.MSA = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.MLP = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=mlp_dim, out_features=embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches + 1, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches + 1, embed_dim).
        """
        x = x.to(self.device)

        # Apply layer normalization before multihead attention
        n1 = self.norm1(x)

        # Multihead attention with residual connection and dropout
        output, _ = self.MSA(n1, n1, n1)
        x = x + self.dropout1(output)

        # Apply layer normalization before MLP
        n2 = self.norm2(x)

        # MLP with residual connection and dropout
        mlp = self.MLP(n2)
        x = x + self.dropout2(mlp)

        return x

class ViT(nn.Module):
    """
    Vision Transformer (ViT) Model

    This class defines the Vision Transformer model, integrating the embedding layer and multiple transformer encoder layers.

    Args:
        image_size (int): The height and width of the input image.
        patch_size (int): The size of each image patch.
        num_classes (int): The number of output classes.
        embed_dim (int): The dimensionality of the latent space.
        num_layers (int): The number of transformer encoder layers.
        num_heads (int): The number of attention heads in each encoder layer.
        mlp_dim (int): The dimensionality of the feed-forward network.
        dropout (float): The dropout rate.
        channels (int): The number of channels in the input image.
        device (torch.device): The device to run the computations on.
    """

    def __init__(self, image_size, patch_size, num_classes, embed_dim, num_layers, num_heads, mlp_dim, dropout=0.1, channels=3, device=None):
        super(ViT, self).__init__()
        self.device = device if device is not None else torch.device('cpu')

        self.embedding = ViTEmbedding(image_size, patch_size, channels, embed_dim, device)

        self.encoder = nn.ModuleList(
            [TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout, device) for _ in range(num_layers)]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Vision Transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = x.to(self.device)
        x = self.embedding(x)  # (batch_size, num_patches + 1, embed_dim)

        for layer in self.encoder:
            x = layer(x)

        cls_token = x[:, 0]  # (batch_size, embed_dim)
        x = self.mlp_head(cls_token)  # (batch_size, num_classes)

        return x