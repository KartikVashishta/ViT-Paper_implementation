import os
import time
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics
from pytorch_lightning import LightningModule, Trainer, Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
import wandb
from torch.utils.data import DataLoader

class LiT(LightningModule):
    """
    LightningModule for training Vision Transformers, Hybrid Models, and ResNets.

    Args:
        model (nn.Module): Model to be trained.
        num_classes (int): Number of output classes.
        lr (float): Learning rate for the optimizer.
        betas (tuple): Betas for the Adam optimizer.
        clip_val (float): Gradient clipping value.
    """

    def __init__(self, model: nn.Module, num_classes: int, lr: float = 1e-3, betas: tuple = (0.9, 0.999), clip_val: float = 1.0):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.betas = betas
        self.clip_val = clip_val

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.val_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LightningModule.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step for the LightningModule.

        Args:
            batch (tuple): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss tensor.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.train_accuracy(y_hat, y)

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True, logger=True)

        # Log gradient norm
        if (batch_idx + 1) % 100 == 0:  # Log every 100 steps to reduce overhead
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_val)
            self.log('grad_norm', grad_norm, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the LightningModule.

        Args:
            batch (tuple): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss tensor.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.val_accuracy(y_hat, y)

        # Update and log metrics
        precision = self.val_precision(y_hat, y)
        recall = self.val_recall(y_hat, y)
        f1 = self.val_f1(y_hat, y)
        self.val_confusion_matrix(y_hat, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_precision', precision, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_recall', recall, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch to log the confusion matrix.
        """
        # Log confusion matrix at the end of each validation epoch
        confusion_matrix = self.val_confusion_matrix.compute().cpu().numpy()
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=list(range(self.num_classes)),
            preds=list(range(self.num_classes)),
            class_names=[str(i) for i in range(self.num_classes)]
        )})

    def configure_optimizers(self):
        """
        Configures the optimizers and learning rate schedulers.

        Returns:
            dict: Dictionary containing the optimizer and scheduler.
        """
        optimizer = Adam(self.parameters(), lr=self.lr, betas=self.betas)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

    def on_before_optimizer_step(self, optimizer):
        """
        Called before the optimizer step to log gradient statistics.

        Args:
            optimizer (Optimizer): Optimizer being used.
        """
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.log(f'{name}_grad_mean', param.grad.mean(), on_step=True, on_epoch=False, prog_bar=False, logger=True)
                self.log(f'{name}_grad_std', param.grad.std(), on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log(f'{name}_weight_mean', param.data.mean(), on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log(f'{name}_weight_std', param.data.std(), on_step=True, on_epoch=False, prog_bar=False, logger=True)

        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_val)

class TimingCallback(Callback):
    """
    Callback to measure and log inference time.

    This callback logs the inference time for each batch during testing.
    """

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.start = time.time()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        duration = time.time() - self.start
        pl_module.log('inference_time', duration)

def train_model(model: nn.Module, model_ver: str, train_dataloader: DataLoader, test_dataloader: DataLoader, lr: float = 1e-4, epochs: int = 14, num_classes: int = 10, accumulate_grad_batches: int = 8, betas: tuple = (0.9, 0.999), clip_val: float = 1.0):
    """
    Trains the specified model using PyTorch Lightning.

    Args:
        model (nn.Module): Model to be trained.
        model_ver (str): Model version name for logging.
        train_dataloader (DataLoader): DataLoader for training data.
        test_dataloader (DataLoader): DataLoader for testing data.
        lr (float): Learning rate for the optimizer.
        epochs (int): Number of epochs to train for.
        num_classes (int): Number of output classes.
        accumulate_grad_batches (int): Number of batches to accumulate gradients over.
        betas (tuple): Betas for the Adam optimizer.
        clip_val (float): Gradient clipping value.
    """
    wandb.init(project="ViT_paper _implementation")

    lit_model = LiT(model, num_classes=num_classes, lr=lr, betas=betas, clip_val=clip_val)

    checkpoint_dir = f'checkpoints/{model_ver}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    wandb.finish()

    logger = WandbLogger(project="ViT_paper _implementation", name=model_ver)

    callbacks = [
        TQDMProgressBar(),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min',
        ),
        TimingCallback(),
    ]

    precision = '16-mixed' if torch.cuda.is_available() else '32'

    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        callbacks=callbacks,
        accelerator='gpu',
        devices=1,
        strategy='ddp' if torch.cuda.is_available() else 'auto',
        num_sanity_val_steps=0,
        accumulate_grad_batches=accumulate_grad_batches,
        precision=precision,
        log_every_n_steps=5
        )

    trainer.fit(lit_model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
