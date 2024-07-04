import torch
from torch import nn
from einops import rearrange, repeat

class LayerScale(nn.Module):
    """
    LayerScale applies a learnable scaling parameter to the output of each layer.

    Args:
        dim (int): Dimension of the input tensor.
        init_val (float): Initial value for the scaling parameter.
        inplace (bool): Whether to apply the scaling in-place.

    Returns:
        torch.Tensor: Scaled tensor.
    """
    def __init__(self, dim: int, init_val: float = 1e-5, inplace: bool = False) -> None:
        super(LayerScale, self).__init__()
        self.gamma = nn.Parameter(init_val * torch.ones(dim))
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class FlashAttention(nn.Module):
    """
    FlashAttention is an optimized multi-head attention mechanism.

    Args:
        embed_dim (int): Dimension of the input tensor.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.

    Returns:
        torch.Tensor: Output tensor after applying attention mechanism.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.num_heads, d=self.head_dim)
        attn = torch.einsum('bhid, bhjd -> bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = torch.einsum('bhij, bhjd -> bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj_drop(self.proj(out))

def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """
    DropPath randomly drops a path in the network during training.

    Args:
        x (torch.Tensor): Input tensor.
        drop_prob (float): Dropout probability.
        training (bool): Whether the model is in training mode.

    Returns:
        torch.Tensor: Tensor with dropped paths.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device).floor_()
    return x * mask / keep_prob

class DropPath(nn.Module):
    """
    DropPath module that wraps the drop_path function.

    Args:
        drop_prob (float): Dropout probability.

    Returns:
        torch.Tensor: Tensor with dropped paths.
    """
    def __init__(self, drop_prob: float = None) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

class FeedForward(nn.Module):
    """
    FeedForward network with GELU activation and dropout.

    Args:
        embed_dim (int): Dimension of the input tensor.
        mlp_dim (int): Dimension of the hidden layer in MLP.
        dropout (float): Dropout rate.

    Returns:
        torch.Tensor: Output tensor after applying feedforward network.
    """
    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class EncoderLayer(nn.Module):
    """
    EncoderLayer for the Vision Transformer, includes FlashAttention and FeedForward networks.

    Args:
        embed_dim (int): Dimension of the input tensor.
        num_heads (int): Number of attention heads.
        mlp_dim (int): Dimension of the hidden layer in MLP.
        dropout (float): Dropout rate.
        init_values (float): Initial value for LayerScale.

    Returns:
        torch.Tensor: Output tensor after applying the encoder layer.
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1, init_values: float = 1e-5) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = FlashAttention(embed_dim, num_heads, dropout)
        self.ls1 = LayerScale(embed_dim, init_values)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, mlp_dim, dropout)
        self.ls2 = LayerScale(embed_dim, init_values)
        self.drop_path = DropPath(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x

class ViTOptimized(nn.Module):
    """
    Optimized Vision Transformer (ViT) Model with LayerScale and FlashAttention.

    Args:
        embed_dim (int): Embedding dimension.
        num_classes (int): Number of output classes.
        num_heads (int): Number of attention heads.
        depth (int): Number of transformer layers.
        mlp_dim (int): Dimension of the hidden layer in MLP.
        dropout (float): Dropout rate.
    """
    def __init__(self, embed_dim: int, num_classes: int, num_heads: int, depth: int, mlp_dim: int, dropout: float):
        super(ViTOptimized, self).__init__()
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1025, embed_dim))  # Initial value, but will be adjusted
        self.pos_dropout = nn.Dropout(dropout)

        self.patch_proj = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=16, stride=16)  # Adjust kernel and stride

        self.encoder = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        Initialize the weights of the model.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ViTOptimized model.

        Args:
            x (torch.Tensor): Input tensor, either 4D or 3D.

        Returns:
            torch.Tensor: Output tensor with class scores.
        """
        print(f"ViTOptimized input shape: {x.shape}")

        if x.dim() == 4:
            b, c, h, w = x.shape
            x = self.patch_proj(x)
            print(f"After patch projection shape: {x.shape}")
            x = x.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)
            print(f"After flatten and transpose shape: {x.shape}")
        elif x.dim() == 3:
            print(f"3D input directly passed to encoder layers")

        num_patches = x.shape[1]
        pos_embed = self.pos_embed[:, :num_patches + 1, :]  # Adjust positional embeddings dynamically

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # Shape: (batch_size, num_patches + 1, embed_dim)

        x = x + pos_embed
        x = self.pos_dropout(x)

        for layer in self.encoder:
            x = layer(x)

        x = self.norm(x)
        return self.head(x[:, 0])
