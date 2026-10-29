"""Train one stage of the Comment-7 hierarchical HSTGNN baseline.

The original five-class labels remain unchanged on disk.  This entry point
only remaps the supervised targets selected by the existing masks:

* ``binary_error``: C0 -> 0 and C1--C4 -> 1;
* ``error_subtype``: C1--C4 -> 0--3, while C0 nodes are excluded.

Keeping the remapping in the training module avoids modifying the campaign
data or the main flat-model training path.
"""

import os
import shutil

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch_geometric.loader import DataLoader

from models.models import get_model
from mydatasets.customized_dataset import CustomGraphDataset, DisplayDatasetInfo
from train import GNNLightning
from utils.config import load_config
from utils.utils import get_logger


logger = get_logger("FsimNNHierarchicalTraining")
VALID_LABEL_MODES = {"binary_error", "error_subtype"}


def select_stage_targets(y, mask, label_mode):
    """Return the mask and remapped targets for one hierarchy stage."""
    if label_mode == "binary_error":
        return mask, (y > 0).long()
    if label_mode == "error_subtype":
        stage_mask = mask & (y > 0)
        return stage_mask, y.long() - 1
    raise ValueError(
        f"Unknown hierarchical label mode: {label_mode}. "
        f"Expected one of {sorted(VALID_LABEL_MODES)}"
    )


class HierarchicalGNNLightning(GNNLightning):
    """GNNLightning with target remapping for a two-stage classifier."""

    def __init__(self, *args, label_mode, **kwargs):
        super().__init__(*args, **kwargs)
        if label_mode not in VALID_LABEL_MODES:
            raise ValueError(f"Unsupported label_mode={label_mode}")
        self.label_mode = label_mode

    def _stage_view(self, out, y, mask):
        stage_mask, stage_targets = select_stage_targets(y, mask, self.label_mode)
        return out[stage_mask], stage_targets[stage_mask], stage_mask

    def training_step(self, batch, batch_idx):
        out = self(batch)
        out_masked, y_masked, mask = self._stage_view(
            out, batch["ff"].y, batch["ff"].train_mask
        )
        if y_masked.numel() == 0:
            return None

        loss = self._compute_loss(out_masked, y_masked)
        acc = self.train_acc_metric(out_masked, y_masked)
        self.log("train_loss", loss, batch_size=int(mask.sum().item()))
        self.log("train_acc", acc, batch_size=int(mask.sum().item()))
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        out_masked, y_masked, mask = self._stage_view(
            out, batch["ff"].y, batch["ff"].val_mask
        )
        if y_masked.numel() == 0:
            return None

        loss = self._compute_loss(out_masked, y_masked)
        acc = self.val_acc_metric(out_masked, y_masked)
        prec = self.precision_metric(out_masked, y_masked)
        rec = self.recall_metric(out_masked, y_masked)
        batch_size = int(mask.sum().item())
        self.log("val_loss", loss, batch_size=batch_size)
        self.log("val_acc", acc, batch_size=batch_size)
        self.log("val_prec", prec, batch_size=batch_size)
        self.log("val_rec", rec, batch_size=batch_size)


def main():
    config = load_config()
    logger.info(config)

    manual_seed = config.get("manual_seed")
    if manual_seed is not None:
        logger.info("Seed the RNG for all devices with %s", manual_seed)
        pl.seed_everything(manual_seed)
        torch.manual_seed(manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    task_config = config.get("task", {})
    label_mode = task_config.get("label_mode")
    if label_mode not in VALID_LABEL_MODES:
        raise ValueError(
            "task.label_mode must be 'binary_error' or 'error_subtype'"
        )

    dataset_config = config["dataset"]
    split_mode = dataset_config.get("split_mode", "spatio")
    if split_mode in {"feature_coverage", "grouped_ff_feature_coverage"}:
        split_config = dataset_config.get(split_mode, {})
    else:
        split_config = dataset_config.get("split_config", {})

    trainer_config = config["trainer"]
    model_config = config["model"]
    num_classes = int(model_config["num_classes"])
    time_window_size = int(model_config.get("time_window_size", 60))
    edge_vector_size = int(model_config.get("edge_vector_size", 10))

    dataset_kwargs = {
        "root": dataset_config["root_dir"],
        "train_val_dirs": dataset_config["train_val_dirs"],
        "test_dirs": dataset_config["test_dirs"],
        "split_mode": split_mode,
        "split_config": split_config,
        "max_node_feat_len": time_window_size,
        "max_edge_feat_len": edge_vector_size,
    }
    train_dataset = CustomGraphDataset(phase="train", **dataset_kwargs)
    val_dataset = CustomGraphDataset(phase="val", **dataset_kwargs)
    DisplayDatasetInfo(train_dataset, f"train/{label_mode}")
    DisplayDatasetInfo(val_dataset, f"val/{label_mode}")

    batch_size = int(trainer_config.get("batch_size", 1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer_config = config["optimizer"]
    model = get_model(config)
    loss_config = config.get("loss", {"name": "ce"})
    lightning_model = HierarchicalGNNLightning(
        model=model,
        lr=optimizer_config.get("learning_rate", 0.001),
        weight_decay=optimizer_config.get("weight_decay", 0.00001),
        factor=optimizer_config.get("factor", 0.5),
        patience=optimizer_config.get("patience", 10),
        num_classes=num_classes,
        loss_config=loss_config,
        label_mode=label_mode,
    )

    checkpoint_dir = trainer_config.get("checkpoint_dir", "./checkpoints/")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    early_stopping = EarlyStopping(
        monitor="avg_val_acc",
        patience=int(trainer_config.get("early_stop_patience", 30)),
        min_delta=float(trainer_config.get("early_stop_min_delta", 0.0)),
        mode="max",
        verbose=True,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="avg_val_acc",
        filename="best_model",
        save_top_k=1,
        mode="max",
        verbose=True,
    )
    trainer = pl.Trainer(
        max_epochs=int(trainer_config.get("epochs", 100)),
        check_val_every_n_epoch=int(
            trainer_config.get("validate_after_epochs", 1)
        ),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[early_stopping, checkpoint_callback],
        deterministic=True,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )
    trainer.fit(lightning_model, train_loader, val_loader)


if __name__ == "__main__":
    main()
