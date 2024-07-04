# Configuration for ViT-L/16

config = {
    "model_ver": "ViT-L-224x224-16-7",
    "model_args": {
        "num_classes": 10,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_dim": 3072,
        "dropout": 0.1
    },
    "train_params": {
        "lr": 1e-4,
        "epochs": 7,
        "num_classes": 10,
        "accumulate_grad_batches": 8,
        "betas": (0.9, 0.999),
        "clip_val": 1.0
    },
    "data_params": {
        "image_size": 224,
        "batch_size": 16,
        "num_images_per_class": 100,
        "train_ratio": 0.8
    }
}
