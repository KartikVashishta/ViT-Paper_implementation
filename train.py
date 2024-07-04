# Modify your train.py to use the updated model class correctly

import argparse
import importlib
import os
import torch
from utils.data import get_dataloaders
from utils.training import train_model
from models.vit_optimized import ViTOptimized
from models.hybrid import HybridModel, ExtendedResNet

def load_config(config_path: str) -> dict:
    """
    Loads the configuration from the specified path.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.config

def main(config_path: str):
    """
    Main function to set up and train the model.

    Args:
        config_path (str): Path to the configuration file.
    """
    # Load configuration
    config = load_config(config_path)

    # Prepare data loaders
    data_params = config["data_params"]
    train_dataloader, test_dataloader = get_dataloaders(
        image_size=data_params["image_size"],
        batch_size=data_params["batch_size"],
        num_images_per_class=data_params["num_images_per_class"],
        train_ratio=data_params["train_ratio"]
    )

    # Initialize model
    model_args = config["model_args"]
    if "resnet_args" in model_args:
        resnet = ExtendedResNet(**model_args["resnet_args"])
        vit = ViTOptimized(**model_args["vit_args"])
        model = HybridModel(resnet, vit)
    else:
        model = ViTOptimized(**model_args) if "embed_dim" in model_args else KNOWN_MODELS[config["model_ver"].split("-")[0]](**model_args)

    # Train model
    train_params = config["train_params"]
    train_model(
        model,
        config["model_ver"],
        train_dataloader,
        test_dataloader,
        lr=train_params["lr"],
        epochs=train_params["epochs"],
        num_classes=train_params["num_classes"],
        accumulate_grad_batches=train_params["accumulate_grad_batches"],
        betas=train_params["betas"],
        clip_val=train_params["clip_val"]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with the specified configuration.")
    parser.add_argument("config", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)
