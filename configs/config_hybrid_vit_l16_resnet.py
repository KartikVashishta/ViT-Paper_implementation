# Configuration for Hybrid Model with ViT-L/16 and ResNet

config = {
    "model_ver": "ViT(L)-Hyb-32x32-14",
    "model_args": {
        "resnet_args": {
            "block_units": [3, 4, 6, 3],
            "width_factor": 1,
            "out_channels": 1024
        },
        "vit_args": {
            "num_classes": 10,
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "mlp_dim": 4096,
            "dropout": 0.1
        }
    },
    "train_params": {
        "lr": 1e-4,
        "epochs": 14,
        "num_classes": 10,
        "accumulate_grad_batches": 8,
        "betas": (0.9, 0.999),
        "clip_val": 1.0
    },
    "data_params": {
        "image_size": 32,
        "batch_size": 16,
        "num_images_per_class": 100,
        "train_ratio": 0.8
    }
}