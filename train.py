import os
import shutil
import sys
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import EarlyStopping
from mydatasets.customized_dataset import CustomGraphDataset, DisplayDatasetInfo

from torchmetrics.classification import Accuracy, Precision, Recall

from models.models import get_model
from utils.utils import get_logger
from utils.config import load_config


logger = get_logger('FsimNNTraining')


def _cuda_mem_stats_mib(device=None):
    """Return CUDA memory stats in MiB (allocated/reserved + peaks).
    Note: reserved is PyTorch caching allocator reserved memory (close to nvidia-smi usage).
    """
    if not torch.cuda.is_available():
        return None
    if device is None:
        device = torch.cuda.current_device()
    return {
        "alloc": torch.cuda.memory_allocated(device) / 1024**2,
        "reserved": torch.cuda.memory_reserved(device) / 1024**2,
        "peak_alloc": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved": torch.cuda.max_memory_reserved(device) / 1024**2,
    }


class GPUMemoryMonitor(Callback):
    """Log GPU memory usage + peak via PyTorch.
    
    - alloc:    current tensor memory
    - reserved: caching allocator reserved memory (often matches nvidia-smi more closely)
    - peak_*:   peak since last reset
    """
    def __init__(self, log_every_n_steps=50, sync=False,
                 reset_peak_each_epoch=True, reset_peak_each_step=False):
        self.log_every_n_steps = int(log_every_n_steps)
        self.sync = bool(sync)
        self.reset_peak_each_epoch = bool(reset_peak_each_epoch)
        self.reset_peak_each_step = bool(reset_peak_each_step)

    def _dev_index(self, pl_module):
        if not torch.cuda.is_available():
            return None
        dev = getattr(pl_module, 'device', None)
        if dev is None or dev.type != 'cuda':
            return None
        return dev.index if dev.index is not None else torch.cuda.current_device()

    def _log(self, pl_module, prefix):
        dev = self._dev_index(pl_module)
        if dev is None:
            return
        if self.sync:
            torch.cuda.synchronize(dev)
        s = _cuda_mem_stats_mib(dev)
        if s is None:
            return
        logger.info(
            f"[GPU_MEM]{prefix} "
            f"alloc={s['alloc']:.1f}MiB reserved={s['reserved']:.1f}MiB | "
            f"peak_alloc={s['peak_alloc']:.1f}MiB peak_reserved={s['peak_reserved']:.1f}MiB"
        )

    def on_fit_start(self, trainer, pl_module):
        dev = self._dev_index(pl_module)
        if dev is None:
            return
        torch.cuda.reset_peak_memory_stats(dev)
        self._log(pl_module, " fit_start")

    def on_train_epoch_start(self, trainer, pl_module):
        dev = self._dev_index(pl_module)
        if dev is None:
            return
        if self.reset_peak_each_epoch:
            torch.cuda.reset_peak_memory_stats(dev)
        self._log(pl_module, f" train_epoch_start e={trainer.current_epoch}")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        dev = self._dev_index(pl_module)
        if dev is None or (not self.reset_peak_each_step):
            return
        torch.cuda.reset_peak_memory_stats(dev)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.log_every_n_steps <= 0:
            return
        if (batch_idx % self.log_every_n_steps) == 0:
            self._log(pl_module, f" train_step e={trainer.current_epoch} b={batch_idx}")

    def on_train_epoch_end(self, trainer, pl_module):
        self._log(pl_module, f" train_epoch_end e={trainer.current_epoch}")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log(pl_module, f" val_epoch_end e={trainer.current_epoch}")

class GNNLightning(pl.LightningModule):
    def __init__(
        self,
        model,
        lr=1e-3,
        weight_decay=5e-4,
        factor=0.5,
        patience=10,
        num_classes=5,
        loss_config=None,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.factor = factor
        self.patience = patience
        self.best_val_acc = 0.0
        self.num_classes = num_classes
        self.loss_config = loss_config or {"name": "ce"}
        self.loss_name = self.loss_config.get("name", "ce")

        class_weights = self.loss_config.get("class_weights", None)
        if class_weights is not None:
            if len(class_weights) != num_classes:
                raise ValueError(
                    f"loss.class_weights length {len(class_weights)} does not match num_classes={num_classes}"
                )
            class_weights = torch.tensor(class_weights, dtype=torch.float)
        else:
            class_weights = torch.ones(num_classes, dtype=torch.float)
        self.register_buffer("class_weights", class_weights)

        # 多分类指标
        self.train_acc_metric = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc_metric   = Accuracy(task="multiclass", num_classes=num_classes)
        self.precision_metric = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.recall_metric    = Recall(task="multiclass", num_classes=num_classes, average="macro")
        


    def forward(self, data):
        return self.model(data)

    def _compute_loss(self, out, y):
        if self.loss_name in ("ce", "cross_entropy"):
            return F.cross_entropy(out, y)
        if self.loss_name in ("weighted_ce", "weighted_cross_entropy"):
            return F.cross_entropy(out, y, weight=self.class_weights)
        raise ValueError(f"Unknown loss name: {self.loss_name}")

    def training_step(self, batch, batch_idx):
        out = self(batch)  # [num_ff, num_classes]
        y = batch["ff"].y
        mask = batch["ff"].train_mask

        out_masked = out[mask]
        y_masked = y[mask]

        loss = self._compute_loss(out_masked, y_masked)
        acc = self.train_acc_metric(out_masked, y_masked)
        self.log("train_loss", loss, batch_size=mask.sum().item())
        self.log("train_acc", acc, batch_size=mask.sum().item())
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        y = batch["ff"].y
        mask = batch["ff"].val_mask

        out_masked = out[mask]
        y_masked = y[mask]

        loss = self._compute_loss(out_masked, y_masked)
        acc = self.val_acc_metric(out_masked, y_masked)
        prec = self.precision_metric(out_masked, y_masked)
        rec = self.recall_metric(out_masked, y_masked)
        
        self.log("val_loss", loss, batch_size=mask.sum().item())
        self.log("val_acc", acc, batch_size=mask.sum().item())
        self.log("val_prec", prec, batch_size=mask.sum().item())
        self.log("val_rec", rec, batch_size=mask.sum().item())
        
    def on_validation_epoch_end(self):
        """Runs once at the end of the validation epoch."""
        # Get all logged values from `self.trainer.callback_metrics`
        avg_loss = self.trainer.callback_metrics["val_loss"]
        avg_acc = self.trainer.callback_metrics["val_acc"]
        logger.info(f'Validation finished. Loss: {avg_loss.item()}, Acc: {avg_acc.item()}')

        if avg_acc.item() > self.best_val_acc:
            self.best_val_acc = avg_acc.item()
            logger.info(f'New best validation accuracy: {self.best_val_acc}')

        # Log final metrics for the epoch
        self.log("avg_val_loss", avg_loss)
        self.log("avg_val_acc", avg_acc)
        
        

    def test_step(self, batch, batch_idx):
        out = self(batch)

    def accuracy(self, out, y, mask):
        pred = out.argmax(dim=1)
        correct = (pred[mask] == y[mask]).sum().float()
        return correct / mask.sum()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=self.factor, patience=self.patience, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "avg_val_acc"}



def main():
    # Load and log experiment configuration
    config = load_config()
    logger.info(config)

    # Set random seed
    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        pl.seed_everything(manual_seed)
        torch.manual_seed(manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    
    dataset_config = config["dataset"]
    root_dir = dataset_config["root_dir"]
    train_val_dirs = dataset_config["train_val_dirs"]
    test_dirs = dataset_config["test_dirs"]
    split_mode = dataset_config.get("split_mode", "spatio")
    if split_mode in {"feature_coverage", "grouped_ff_feature_coverage"}:
        split_config = dataset_config.get(split_mode, {})
    else:
        split_config = dataset_config.get("split_config", {})
    
    trainer_config = config["trainer"]
    batch_size = trainer_config.get("batch_size", 1)
    model_config = config["model"]
    num_classes = model_config.get("num_classes", 5)
    edge_vector_size = model_config.get("edge_vector_size", 10)
    time_window_size = model_config.get("time_window_size", 60)
    train_dataset = CustomGraphDataset(root_dir, train_val_dirs, test_dirs, split_mode=split_mode, split_config=split_config, phase="train", max_node_feat_len=time_window_size, max_edge_feat_len=edge_vector_size)
    val_dataset = CustomGraphDataset(root_dir, train_val_dirs, test_dirs, split_mode=split_mode, split_config=split_config, phase="val", max_node_feat_len=time_window_size, max_edge_feat_len=edge_vector_size)
    DisplayDatasetInfo(train_dataset, "train")
    DisplayDatasetInfo(val_dataset, "val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    checkpoint_dir = trainer_config.get("checkpoint_dir", "./checkpoints/")
    epochs = trainer_config.get("epochs", 100)
    validate_after_epochs = trainer_config.get("validate_after_epochs", 1)
    early_stop_patience = trainer_config.get("early_stop_patience", 30)

    optimizer_config = config["optimizer"]
    learning_rate = optimizer_config.get("learning_rate", "0.001")
    weight_decay = optimizer_config.get("weight_decay", "0.00001")
    factor = optimizer_config.get("factor", "0.5")
    patience = optimizer_config.get("patience", "10")
    model = get_model(config)
    logger.info(f'The details of the model: {model}')
    loss_config = config.get("loss", {"name": "ce"})
    logger.info(f"Loss configuration: {loss_config}")
    lightning_model = GNNLightning(
        model=model,
        lr=learning_rate,
        weight_decay=weight_decay,
        factor=factor,
        patience=patience,
        num_classes=num_classes,
        loss_config=loss_config,
    )

    # Early stopping
    early_stopping = EarlyStopping(monitor="avg_val_acc", patience=early_stop_patience, mode="max", verbose=True)
    
    # Ensure the directory is clean before starting a new training session
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)  # Deletes old model files
    os.makedirs(checkpoint_dir, exist_ok=True)  # Creates a fresh directory
    
    # Define ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="avg_val_acc",  # Metric to monitor
        filename="best_model",  # Name of the saved file
        save_top_k=1,           # Save only the best model
        mode="max",             # "min" for loss (lower is better), "max" for accuracy
        verbose=True
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,  # Total number of epochs
        check_val_every_n_epoch=validate_after_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[
            early_stopping,
            checkpoint_callback,
            # GPUMemoryMonitor(
            #     log_every_n_steps=trainer_config.get("mem_log_every_n_steps", 50),
            #     sync=trainer_config.get("mem_log_sync", False),
            #     reset_peak_each_epoch=trainer_config.get("mem_reset_peak_each_epoch", True),
            #     reset_peak_each_step=trainer_config.get("mem_reset_peak_each_step", False),
            # ),
        ],
        deterministic=True,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1
    )

    # Train
    trainer.fit(lightning_model, train_loader, val_loader)





if __name__ == '__main__':
    main()
