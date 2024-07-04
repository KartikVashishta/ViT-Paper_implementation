# Configuration for ResNet Model

config = {
    "model_ver": "BiT-M-R50x1-224x224-7",
    "model_args": {
        "head_size": 10
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