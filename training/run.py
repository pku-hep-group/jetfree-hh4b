#!/usr/bin/env python3

import argparse
import copy
import json
import logging
import os
import sys
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

REPO_ROOT = Path(__file__).resolve().parent
WEAVER_ROOT = REPO_ROOT / "weaver-core"
if str(WEAVER_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAVER_ROOT))

from weaver.utils.dataset import SimpleIterDataset  # noqa: E402
from weaver.utils.nn.optimizer.ranger import Ranger  # noqa: E402
from model import ParticleTransformer  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("run")


EXPERIMENT_NAME = "hh4b_resolved_newsp4_allparts_nosel_138clswtop.test.noweights.ddp4-bs512-lr2e-3"


@dataclass
class ModelConfig:
    input_dim: int = 19
    num_classes: int = 138
    pair_input_type: str = "pp"
    pair_input_dim: int = 4
    pair_extra_dim: int = 0
    remove_self_pair: bool = False
    use_pre_activation_pair: bool = True
    embed_dims: tuple[int, ...] = (256, 1024, 256)
    pair_embed_dims: tuple[int, ...] = (64, 64, 64)
    num_heads: int = 16
    num_layers: int = 8
    num_cls_layers: int = 2
    block_params: dict[str, Any] = field(default_factory=dict)
    cls_block_params: dict[str, Any] = field(
        default_factory=lambda: {"dropout": 0.0, "attn_dropout": 0.0, "activation_dropout": 0.0}
    )
    fc_params: tuple = ((1024, 0.1),)
    activation: str = "gelu"
    weight_init: str = "moco"
    fix_init: bool = True
    trim: bool = True
    for_inference: bool = False
    use_amp: bool = False


@dataclass
class OptimizerConfig:
    lr: float = 2e-3


@dataclass
class SchedulerConfig:
    decay_fraction: float = 0.3
    final_lr_ratio: float = 0.01


@dataclass
class DataConfig:
    data_config_path: str = str(REPO_ROOT / "data" / "hh4b_resolved_newsp4_allparts_nosel_138clswtop.yaml")
    dataset_dir: str = str(REPO_ROOT / "datasets")
    fetch_by_files: bool = False
    fetch_step: float = 1.0
    file_fraction: float = 1.0
    data_fraction: float = 1.0
    data_split_num: int = 500
    in_memory: bool = False
    num_workers: int = 5
    batch_size: int = 512
    train_samples_per_epoch_total: int = 10000 * 1024
    val_samples_per_epoch_total: int = 2500 * 1024


@dataclass
class TrainerConfig:
    num_epochs: int = 80
    ngpus: int = 4
    accelerator: str = "gpu"
    precision: str = "16-mixed"
    seed: int = 2026
    log_every_n_steps: int = 20
    gradient_clip_val: float = 0.0


MODEL_CFG = ModelConfig()
OPTIM_CFG = OptimizerConfig()
SCHED_CFG = SchedulerConfig()
DATA_CFG = DataConfig()
TRAINER_CFG = TrainerConfig()


def get_output_dir() -> Path:
    return Path("outputs") / EXPERIMENT_NAME


def get_tensorboard_dir() -> Path:
    return Path("tensorboard") / EXPERIMENT_NAME


def resolve_run_profile(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "run_profile": "smoke-test" if args.smoke_test else "full",
        "num_epochs": 1 if args.smoke_test else TRAINER_CFG.num_epochs,
    }


def to_python_scalar(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-file Lightning runner for the 138-class ParticleTransformerHH4b model."
    )
    parser.add_argument("--mode", choices=("train", "predict", "export"), default="train")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Checkpoint used for prediction, ONNX export, or resumed training.")
    parser.add_argument("--predict-output", type=str, default=None, help="Prediction output path. Supports .root and .parquet; named sample groups get a suffix.")
    parser.add_argument("--export-output", type=str, default=None, help="ONNX output path used by --mode export (default: export/<experiment>/model.onnx).")
    parser.add_argument("--stat-only", action="store_true", help="Print model, params and FLOPs without training.")
    parser.add_argument("--smoke-test", action="store_true", help="Run one train/val batch, or one prediction batch per sample group.")
    parser.add_argument("--devices", type=int, default=None, help="Override the number of GPUs defined in run.py.")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None, help="Override the default per-device batch size (512).")
    parser.add_argument("--start-lr", type=float, default=None, help="Override the default learning rate (2e-3).")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    pl.seed_everything(seed, workers=True)


def build_file_specs(cfg: DataConfig) -> list[str]:
    dataset_dir = Path(cfg.dataset_dir)
    specs = [
        f"gghh:{dataset_dir / 'HH4b_2HDM_H3VAR_H1H2_40to200_merged_ntuple'}/*.root",
        f"qcd:{dataset_dir / 'QCD_DelphesHH4JTrig_merged_ntuple'}/*.root",
        f"ttbar:{dataset_dir / 'TTbar_ntuple'}/*.root",
    ]
    # specs = [
    #     f"gghh:{dataset_dir / 'HH4b_2HDM_H3VAR_H1H2_40to200_merged_ntuple' / 'HH4b_2HDM_H3VAR_H1H2_40to200_ntuple_id3140-3149.root'}",
    # ]
    return specs


def to_file_dict(file_specs: Sequence[str], split_by_rank: bool) -> dict[str, list[str]]:
    import glob

    file_dict: dict[str, list[str]] = {}
    for spec in file_specs:
        if ":" in spec:
            group, pattern = spec.split(":", 1)
        else:
            group, pattern = "_", spec
        file_dict.setdefault(group, []).extend(sorted(glob.glob(pattern)))

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if split_by_rank and world_size > 1:
        rank = int(os.environ.get("RANK", "0"))
        split: dict[str, list[str]] = {}
        for group, files in file_dict.items():
            rank_files = files[rank::world_size]
            if not rank_files:
                raise RuntimeError(f"Rank {rank} received no files for group {group}.")
            split[group] = rank_files
        file_dict = split

    return file_dict


def build_prediction_loaders(cfg: DataConfig) -> dict[str, DataLoader]:
    prediction_specs = build_file_specs(cfg)
    file_dict = to_file_dict(prediction_specs, split_by_rank=False)
    loaders = {}
    for group, files in file_dict.items():
        if not files:
            raise RuntimeError(f"No prediction files found for group {group}.")
        dataset = SimpleIterDataset(
            {group: files},
            cfg.data_config_path,
            for_training=False,
            load_range_and_fraction=((0, 1), cfg.data_fraction, cfg.data_split_num),
            file_fraction=cfg.file_fraction,
            fetch_by_files=cfg.fetch_by_files,
            fetch_step=cfg.fetch_step,
            name=f"predict_{group}",
        )
        num_workers = min(cfg.num_workers, len(files))
        loaders[group] = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            drop_last=False,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
    return loaders


def infer_data_config(data_config_path: str):
    dataset = SimpleIterDataset({}, data_config_path, for_training=False, fetch_by_files=True, fetch_step=1)
    return dataset.config


class WeaverDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataConfig, runtime_devices: int, num_nodes: int):
        super().__init__()
        self.cfg = cfg
        self.world_size = max(1, runtime_devices * num_nodes)
        self.file_specs = build_file_specs(cfg)
        self.train_dataset = None
        self.val_dataset = None
        self.data_config = infer_data_config(cfg.data_config_path)
        self.train_steps_per_epoch = max(
            1, cfg.train_samples_per_epoch_total // (cfg.batch_size * self.world_size)
        )
        self.val_steps_per_epoch = max(
            1, cfg.val_samples_per_epoch_total // (cfg.batch_size * self.world_size)
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit", "validate"):
            train_file_dict = to_file_dict(self.file_specs, split_by_rank=True)
            val_file_dict = to_file_dict(self.file_specs, split_by_rank=True)
            self.train_dataset = SimpleIterDataset(
                train_file_dict,
                self.cfg.data_config_path,
                for_training=True,
                load_range_and_fraction=((0, 0.8), self.cfg.data_fraction, self.cfg.data_split_num),
                file_fraction=self.cfg.file_fraction,
                fetch_by_files=self.cfg.fetch_by_files,
                fetch_step=self.cfg.fetch_step,
                infinity_mode=True,
                in_memory=self.cfg.in_memory,
                name=f"train_rank{os.environ.get('LOCAL_RANK', '0')}",
            )
            self.val_dataset = SimpleIterDataset(
                val_file_dict,
                self.cfg.data_config_path,
                for_training=True,
                load_range_and_fraction=((0.8, 1), self.cfg.data_fraction, self.cfg.data_split_num),
                file_fraction=self.cfg.file_fraction,
                fetch_by_files=self.cfg.fetch_by_files,
                fetch_step=self.cfg.fetch_step,
                infinity_mode=True,
                in_memory=self.cfg.in_memory,
                name=f"val_rank{os.environ.get('LOCAL_RANK', '0')}",
            )
            self.data_config = self.train_dataset.config

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
        )

def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=1) == labels).float().mean()


def binary_roc_metrics(scores: np.ndarray, labels: np.ndarray, background_index: int) -> dict[str, Any]:
    from sklearn.metrics import auc, roc_curve

    selected = ((labels >= 0) & (labels < 136)) | (labels == background_index)
    binary_labels = labels[selected] < 136
    selected_scores = scores[selected]
    if not np.any(binary_labels) or np.all(binary_labels):
        return {}

    fpr, tpr, _ = roc_curve(binary_labels, selected_scores)
    signal_efficiencies = (0.8, 0.5, 0.3, 0.2, 0.1)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "auc": float(auc(fpr, tpr)),
        "background_rejection": {
            efficiency: float(np.interp(efficiency, tpr, 1.0 / np.maximum(fpr, 1e-10)))
            for efficiency in signal_efficiencies
        },
    }


class ParticleTransformerHH4b(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.mod = ParticleTransformer(
            input_dim=cfg.input_dim,
            num_classes=cfg.num_classes,
            pair_input_type=cfg.pair_input_type,
            pair_input_dim=cfg.pair_input_dim,
            pair_extra_dim=cfg.pair_extra_dim,
            remove_self_pair=cfg.remove_self_pair,
            use_pre_activation_pair=cfg.use_pre_activation_pair,
            embed_dims=cfg.embed_dims,
            pair_embed_dims=cfg.pair_embed_dims,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            num_cls_layers=cfg.num_cls_layers,
            block_params=cfg.block_params,
            cls_block_params=cfg.cls_block_params,
            fc_params=cfg.fc_params,
            activation=cfg.activation,
            weight_init=cfg.weight_init,
            fix_init=cfg.fix_init,
            trim=cfg.trim,
            for_inference=cfg.for_inference,
            use_amp=cfg.use_amp,
        )

    def forward(self, points, features, lorentz_vectors, mask):
        del points
        return self.mod(features, v=lorentz_vectors, mask=mask)


class ParticleTransformerHH4bONNXWrapper(nn.Module):
    """Match the inference path exported by the original Weaver network wrapper."""

    def __init__(self, model: ParticleTransformerHH4b):
        super().__init__()
        self.mod = model.mod

    def forward(self, points, features, lorentz_vectors, mask):
        del points
        x, padding_mask = self.mod._forward_encoder(features, v=lorentz_vectors, mask=mask)
        x_cls = self.mod._forward_aggregator(x, padding_mask)
        if self.mod.fc is None:
            return x_cls
        return torch.softmax(self.mod.fc(x_cls), dim=1)


class ParticleTransformerHH4bLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_cfg: ModelConfig,
        optim_cfg: OptimizerConfig,
        sched_cfg: SchedulerConfig,
    ):
        super().__init__()
        if isinstance(model_cfg, Mapping):
            model_cfg = ModelConfig(**model_cfg)
        if isinstance(optim_cfg, Mapping):
            optim_cfg = OptimizerConfig(**optim_cfg)
        if isinstance(sched_cfg, Mapping):
            sched_cfg = SchedulerConfig(**sched_cfg)
        self.save_hyperparameters(
            {
                "model_cfg": asdict(model_cfg),
                "optim_cfg": asdict(optim_cfg),
                "sched_cfg": asdict(sched_cfg),
            }
        )
        self.model_cfg = copy.deepcopy(model_cfg)
        self.optim_cfg = copy.deepcopy(optim_cfg)
        self.sched_cfg = copy.deepcopy(sched_cfg)
        self.model = ParticleTransformerHH4b(self.model_cfg)
        self.loss_fn = nn.CrossEntropyLoss()
        self._roc_outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._epoch_lr: Optional[float] = None

    def forward(self, points, features, lorentz_vectors, mask):
        return self.model(points, features, lorentz_vectors, mask)

    def _shared_step(self, batch, stage: str):
        x, y, _ = batch
        labels = y["truth_label"].long()
        logits = self(x["pf_points"], x["pf_features"], x["pf_vectors"], x["pf_mask"])
        loss = self.loss_fn(logits, labels)
        acc = accuracy_from_logits(logits, labels)
        if stage == "val":
            probabilities = torch.softmax(logits.detach().float(), dim=1)
            hh4b_score = probabilities[:, :136].sum(dim=1)
            qcd_score = probabilities[:, 136]
            ttbar_score = probabilities[:, 137]
            qcd_discriminant = hh4b_score / (hh4b_score + qcd_score).clamp_min(1e-12)
            ttbar_discriminant = hh4b_score / (hh4b_score + ttbar_score).clamp_min(1e-12)
            self._roc_outputs.append(
                (qcd_discriminant.cpu(), ttbar_discriminant.cpu(), labels.detach().cpu())
            )
        if stage == "train":
            self.log("Loss/train_step", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log(f"Loss/{stage}_epoch", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False, sync_dist=True)
        self.log(f"Acc/{stage}_epoch", acc, on_step=False, on_epoch=True, prog_bar=True, logger=False, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        del batch_idx
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        del batch_idx
        return self._shared_step(batch, "val")

    def _write_epoch_scalars(self, names: Sequence[str]) -> None:
        if not self.trainer.is_global_zero or self.trainer.sanity_checking:
            return
        writer = self.logger.experiment
        for name in names:
            value = to_python_scalar(self.trainer.callback_metrics.get(name))
            if value is not None:
                writer.add_scalar(name, value, self.current_epoch)

    def on_train_epoch_start(self) -> None:
        self._epoch_lr = float(self.trainer.optimizers[0].param_groups[0]["lr"])

    def on_train_epoch_end(self) -> None:
        self._write_epoch_scalars(("Loss/train_epoch", "Acc/train_epoch"))
        if self.trainer.is_global_zero and self._epoch_lr is not None:
            self.logger.experiment.add_scalar("LR/train_epoch", self._epoch_lr, self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        self._roc_outputs.clear()

    def _gather_roc_outputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        local_qcd_scores = torch.cat([item[0] for item in self._roc_outputs])
        local_ttbar_scores = torch.cat([item[1] for item in self._roc_outputs])
        local_labels = torch.cat([item[2] for item in self._roc_outputs])
        local = torch.stack((local_qcd_scores, local_ttbar_scores, local_labels.float()), dim=1).to(self.device)

        local_size = torch.tensor([local.shape[0]], device=self.device)
        sizes = self.all_gather(local_size).reshape(-1).long()
        max_size = int(sizes.max().item())
        if local.shape[0] < max_size:
            local = torch.nn.functional.pad(local, (0, 0, 0, max_size - local.shape[0]))
        gathered = self.all_gather(local).reshape(-1, max_size, 3)
        chunks = [gathered[index, :size] for index, size in enumerate(sizes.tolist())]
        combined = torch.cat(chunks).cpu().numpy()
        return combined[:, 0], combined[:, 1], combined[:, 2].astype(np.int64)

    def _log_roc_metrics(self) -> None:
        if self.trainer.sanity_checking or not self._roc_outputs:
            return
        qcd_scores, ttbar_scores, labels = self._gather_roc_outputs()
        self._roc_outputs.clear()
        if not self.trainer.is_global_zero:
            return

        writer = self.logger.experiment
        import matplotlib.pyplot as plt

        comparisons = (("QCD", qcd_scores, 136), ("TTbar", ttbar_scores, 137))
        for background_name, scores, background_index in comparisons:
            metrics = binary_roc_metrics(scores, labels, background_index)
            if not metrics:
                continue

            writer.add_scalar(f"AUC_HH4b_vs_{background_name}/val_epoch", metrics["auc"], self.current_epoch)
            for efficiency, rejection in metrics["background_rejection"].items():
                efficiency_tag = str(efficiency).replace(".", "p")
                writer.add_scalar(
                    f"BkgRej{efficiency_tag}_HH4b_vs_{background_name}/val_epoch",
                    rejection,
                    self.current_epoch,
                )

            figure, axis = plt.subplots(figsize=(5, 5))
            axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
            axis.plot(
                metrics["tpr"],
                metrics["fpr"],
                label=f"HH4b vs {background_name} (AUC={metrics['auc']:.4f})",
            )
            axis.set_xlabel("True positive rate (signal efficiency)")
            axis.set_ylabel("False positive rate (background efficiency)")
            axis.set_xlim(0, 1)
            axis.set_ylim(1e-4, 1)
            axis.set_yscale("log")
            axis.legend()
            figure.tight_layout()
            writer.add_figure(f"ROC_HH4b_vs_{background_name}/val_epoch", figure, self.current_epoch)
            plt.close(figure)

    def on_validation_epoch_end(self) -> None:
        self._write_epoch_scalars(("Loss/val_epoch", "Acc/val_epoch"))
        self._log_roc_metrics()

    def configure_optimizers(self):
        optimizer = Ranger(self.parameters(), lr=self.optim_cfg.lr)
        decay_epochs = max(1, int(TRAINER_CFG.num_epochs * self.sched_cfg.decay_fraction))
        first_decay_epoch = TRAINER_CFG.num_epochs - decay_epochs
        gamma = self.sched_cfg.final_lr_ratio ** (1.0 / decay_epochs) 
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(range(first_decay_epoch, TRAINER_CFG.num_epochs)),
            gamma=gamma,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }


class BestValMetricTracker(Callback):
    def __init__(self):
        super().__init__()
        self.best_metrics: dict[str, Any] = {
            "best_val_loss": None,
            "best_val_acc": None,
            "best_epoch": None,
            "best_global_step": None,
        }

    def state_dict(self) -> dict[str, Any]:
        return {"best_metrics": self.best_metrics.copy()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.best_metrics.update(state_dict.get("best_metrics", {}))

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        del pl_module
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        val_loss = to_python_scalar(metrics.get("Loss/val_epoch"))
        val_acc = to_python_scalar(metrics.get("Acc/val_epoch"))
        if val_loss is None or val_acc is None:
            return
        best_val_acc = self.best_metrics["best_val_acc"]
        if best_val_acc is None or val_acc > best_val_acc:
            self.best_metrics = {
                "best_val_loss": val_loss,
                "best_val_acc": val_acc,
                "best_epoch": int(trainer.current_epoch),
                "best_global_step": int(trainer.global_step),
            }


def effective_devices(args: argparse.Namespace) -> int:
    devices = args.devices if args.devices is not None else TRAINER_CFG.ngpus
    if TRAINER_CFG.accelerator == "gpu" and torch.cuda.is_available():
        return max(1, min(devices, torch.cuda.device_count()))
    return 1


def should_enable_model_amp(mode: str, cuda_available: bool, precision: str) -> bool:
    return mode in ("train", "predict") and cuda_available and precision != "32-true"


def build_trainer(
    args: argparse.Namespace,
    runtime_devices: int,
    num_epochs: int,
    datamodule: WeaverDataModule,
) -> pl.Trainer:
    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = Path("tensorboard")
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    logger = TensorBoardLogger(save_dir=str(tensorboard_dir), name="", version=EXPERIMENT_NAME)
    best_val_tracker = BestValMetricTracker()
    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="epoch{epoch:02d}-val_acc{Acc/val_epoch:.4f}",
            monitor="Acc/val_epoch",
            mode="max",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        best_val_tracker,
    ]

    accelerator = TRAINER_CFG.accelerator if runtime_devices > 0 and torch.cuda.is_available() else "cpu"
    strategy = "ddp" if accelerator == "gpu" and runtime_devices > 1 else "auto"
    precision = TRAINER_CFG.precision if accelerator == "gpu" else "32-true"

    return pl.Trainer(
        accelerator=accelerator,
        devices=runtime_devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=precision,
        max_epochs=num_epochs,
        default_root_dir=str(output_dir),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=TRAINER_CFG.log_every_n_steps,
        gradient_clip_val=TRAINER_CFG.gradient_clip_val,
        limit_train_batches=1 if args.smoke_test else datamodule.train_steps_per_epoch,
        limit_val_batches=1 if args.smoke_test else datamodule.val_steps_per_epoch,
    )


def print_run_summary(runtime_devices: int, num_nodes: int, run_profile: dict[str, Any]) -> None:
    world_size = runtime_devices * num_nodes
    steps_per_epoch = max(1, DATA_CFG.train_samples_per_epoch_total // (DATA_CFG.batch_size * world_size))
    val_steps_per_epoch = max(1, DATA_CFG.val_samples_per_epoch_total // (DATA_CFG.batch_size * world_size))
    total_steps = steps_per_epoch * run_profile["num_epochs"]
    LOGGER.info("Experiment name: %s", EXPERIMENT_NAME)
    LOGGER.info("Output dir: %s", get_output_dir())
    LOGGER.info("TensorBoard dir: %s", get_tensorboard_dir())
    LOGGER.info("Run profile: %s", run_profile["run_profile"])
    LOGGER.info("Data config yaml: %s", DATA_CFG.data_config_path)
    LOGGER.info("Dataset dir: %s", DATA_CFG.dataset_dir)
    LOGGER.info("Train/val ranges: [0, 0.8) / [0.8, 1]")
    LOGGER.info("Interval splits per worker: %s", DATA_CFG.data_split_num)
    LOGGER.info("Optimizer: Ranger")
    LOGGER.info("LR: %s", OPTIM_CFG.lr)
    LOGGER.info("Scheduler: flat+decay over the final %.0f%% of epochs", SCHED_CFG.decay_fraction * 100)
    LOGGER.info("Batch size per process: %s", DATA_CFG.batch_size)
    LOGGER.info("Devices per node: %s", runtime_devices)
    LOGGER.info("Nodes: %s", num_nodes)
    LOGGER.info("World size: %s", world_size)
    LOGGER.info("Epochs: %s", run_profile["num_epochs"])
    LOGGER.info("Train steps per epoch: %s", steps_per_epoch)
    LOGGER.info("Val steps per epoch: %s", val_steps_per_epoch)
    LOGGER.info("Total train steps: %s", total_steps)


def build_model_for_stats(data_config) -> ParticleTransformerHH4b:
    model_cfg = copy.deepcopy(MODEL_CFG)
    model_cfg.input_dim = len(data_config.input_dicts["pf_features"])
    model_cfg.for_inference = False
    model = ParticleTransformerHH4b(model_cfg)
    model.eval()
    return model


def run_stat_only() -> None:
    data_config = infer_data_config(DATA_CFG.data_config_path)
    model = build_model_for_stats(data_config)
    print(model)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"\nTotal params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        print("ptflops is not available in the active environment.")
        return

    input_shapes = tuple(tuple(data_config.input_shapes[name][1:]) for name in data_config.input_names)

    def input_constructor(input_res):
        dummy = {}
        for name, shape in zip(data_config.input_names, input_res):
            dummy[name] = torch.ones((1,) + tuple(shape), dtype=torch.float32)
        return {
            "points": dummy["pf_points"],
            "features": dummy["pf_features"],
            "lorentz_vectors": dummy["pf_vectors"],
            "mask": dummy["pf_mask"],
        }

    macs, params = get_model_complexity_info(
        model,
        input_shapes,
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True,
        input_constructor=input_constructor,
    )
    print(f"ptflops params: {params}")
    print(f"ptflops MACs/FLOPs: {macs}")


def default_checkpoint() -> Optional[str]:
    candidate = get_output_dir() / "checkpoints" / "last.ckpt"
    return str(candidate) if candidate.exists() else None


def default_predict_output() -> Path:
    return Path("predict") / EXPERIMENT_NAME / "pred.root"


def default_export_output() -> Path:
    return Path("export") / EXPERIMENT_NAME / "model.onnx"


def load_prediction_weights(module: ParticleTransformerHH4bLightningModule, checkpoint_path: str) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload).__name__}")

    state_dict = payload.get("state_dict", payload.get("model_state_dict", payload))
    state_dict = {
        (key.removeprefix("module.")): value
        for key, value in state_dict.items()
    }
    try:
        module.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError as lightning_error:
        try:
            module.model.load_state_dict(state_dict, strict=True)
            return
        except RuntimeError as model_error:
            raise RuntimeError(
                f"Checkpoint '{checkpoint_path}' matches neither the Lightning module nor the raw model.\n"
                f"Lightning load error: {lightning_error}\nRaw model load error: {model_error}"
            ) from model_error


def export_onnx(
    lit_model: ParticleTransformerHH4bLightningModule,
    checkpoint_path: str,
    output_path: Path,
    data_config,
) -> None:
    if output_path.suffix.lower() != ".onnx":
        raise ValueError(f"ONNX output path must end with .onnx: {output_path}")

    load_prediction_weights(lit_model, checkpoint_path)
    model = ParticleTransformerHH4bONNXWrapper(lit_model.model).cpu().eval()
    input_names = list(data_config.input_names)
    output_names = ["softmax"]
    input_shapes = {
        name: (1,) + tuple(data_config.input_shapes[name][1:])
        for name in input_names
    }
    inputs = tuple(torch.ones(input_shapes[name], dtype=torch.float32) for name in input_names)
    dynamic_axes = {
        **{name: {0: "N", 2: f"n_{name.split('_')[0]}"} for name in input_names},
        "softmax": {0: "N"},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        dynamo=False,
    )
    LOGGER.info("ONNX model saved to %s", output_path)

    preprocessing_path = output_path.parent / "preprocess.json"
    data_config.export_json(str(preprocessing_path))
    LOGGER.info("Preprocessing parameters saved to %s", preprocessing_path)


def prediction_output_path(base_path: Path, group: str) -> Path:
    if not group:
        return base_path
    return base_path.with_name(f"{base_path.stem}_{group}{base_path.suffix}")


def write_prediction_output(
    output_path: Path,
    scores: np.ndarray,
    labels: dict[str, np.ndarray],
    observers: dict[str, np.ndarray],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".root":
        from weaver.utils.data.fileio import _write_root

        output = {"cls_index": labels["truth_label"]}
        output.update({f"score_{index}": scores[:, index] for index in range(scores.shape[1])})
        for values in (labels, observers):
            for name, value in values.items():
                if name == "truth_label":
                    continue
                if value.ndim != 1:
                    LOGGER.warning("Ignoring non-1D prediction field %s with shape %s.", name, value.shape)
                    continue
                output[name] = value
        _write_root(str(output_path), output)
        return

    if suffix == ".parquet":
        import awkward as ak

        output = {"scores": scores, **labels, **observers}
        ak.to_parquet(ak.Array(output), str(output_path), compression="LZ4", compression_level=4)
        return

    raise ValueError(f"Unsupported prediction output format '{suffix}'; use .root or .parquet.")


def run_prediction(
    lit_model: ParticleTransformerHH4bLightningModule,
    checkpoint_path: str,
    output_path: Path,
    runtime_devices: int,
    smoke_test: bool,
) -> None:
    load_prediction_weights(lit_model, checkpoint_path)
    loaders = build_prediction_loaders(DATA_CFG)

    if torch.cuda.is_available() and runtime_devices > 0:
        device = torch.device("cuda:0")
        model = lit_model.model.to(device)
        if runtime_devices > 1:
            model = nn.DataParallel(model, device_ids=list(range(runtime_devices)))
    else:
        device = torch.device("cpu")
        model = lit_model.model.to(device)
    model.eval()
    use_amp = lit_model.model_cfg.use_amp

    from tqdm.auto import tqdm

    for group, loader in loaders.items():
        score_batches = []
        label_batches: dict[str, list[np.ndarray]] = defaultdict(list)
        observer_batches: dict[str, list[np.ndarray]] = defaultdict(list)
        num_correct = 0
        num_events = 0
        with torch.inference_mode():
            for batch_index, (x, y, observers) in enumerate(tqdm(loader, desc=f"Predict {group}")):
                inputs = [
                    x[name].to(device, non_blocking=True)
                    for name in ("pf_points", "pf_features", "pf_vectors", "pf_mask")
                ]
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    logits = model(*inputs)
                probabilities = torch.softmax(logits.float(), dim=1).cpu()
                score_batches.append(probabilities.numpy())

                for name, value in y.items():
                    label_batches[name].append(value.detach().cpu().numpy())
                for name, value in observers.items():
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().numpy()
                    observer_batches[name].append(np.asarray(value))

                truth = y["truth_label"].long()
                num_correct += int((probabilities.argmax(dim=1) == truth.cpu()).sum().item())
                num_events += int(truth.shape[0])
                if smoke_test and batch_index == 0:
                    break

        if not score_batches:
            raise RuntimeError(f"Prediction produced no events for group {group}.")
        scores = np.concatenate(score_batches)
        labels = {name: np.concatenate(values) for name, values in label_batches.items()}
        observers = {name: np.concatenate(values) for name, values in observer_batches.items()}
        group_output_path = prediction_output_path(output_path, group)
        write_prediction_output(group_output_path, scores, labels, observers)
        LOGGER.info(
            "Written %d predictions for %s to %s (accuracy %.6f).",
            num_events,
            group,
            group_output_path,
            num_correct / num_events,
        )


def ensure_fresh_experiment_name(args: argparse.Namespace) -> None:
    if args.stat_only or args.mode != "train" or args.ckpt_path is not None:
        return
    output_dir = get_output_dir()
    tensorboard_dir = get_tensorboard_dir()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory '{output_dir}' already exists and is not empty. "
            "Change EXPERIMENT_NAME before launching a new training run."
        )
    if tensorboard_dir.exists() and any(tensorboard_dir.iterdir()):
        raise FileExistsError(
            f"TensorBoard directory '{tensorboard_dir}' already exists and is not empty. "
            "Change EXPERIMENT_NAME before launching a new training run."
        )


def get_best_val_tracker(trainer: pl.Trainer) -> Optional[BestValMetricTracker]:
    for callback in trainer.callbacks:
        if isinstance(callback, BestValMetricTracker):
            return callback
    return None


def write_train_summary(summary: dict[str, Any]) -> None:
    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def print_train_summary(summary: dict[str, Any]) -> None:
    print("---")
    for key in (
        "best_val_loss",
        "best_val_acc",
        "best_epoch",
        "best_global_step",
        "run_profile",
        "training_seconds",
        "total_seconds",
        "peak_vram_mb",
        "num_params_M",
        "num_trainable_params_M",
        "train_steps",
        "train_steps_per_epoch",
        "val_steps_per_epoch",
        "checkpoint_path",
        "experiment_name",
        "output_dir",
        "tensorboard_dir",
    ):
        if key in summary:
            print(f"{key+':':20s} {summary[key]}")


def normalize_checkpoint_path(checkpoint_path: Optional[str]) -> Optional[str]:
    if checkpoint_path is None:
        return None
    ckpt_name = Path(checkpoint_path).name
    return str(get_output_dir() / "checkpoints" / ckpt_name)


def main() -> None:
    t_start_total = time.time()
    args = parse_args()
    if args.batch_size is not None:
        DATA_CFG.batch_size = args.batch_size
    if args.start_lr is not None:
        OPTIM_CFG.lr = args.start_lr
    if args.smoke_test:
        DATA_CFG.num_workers = 0
        if args.mode == "train":
            # Match the per-worker file count of the default 4 GPU x 5 worker run.
            DATA_CFG.file_fraction = 0.05
    torch.set_float32_matmul_precision("high")
    runtime_devices = effective_devices(args)
    run_profile = resolve_run_profile(args)
    seed_everything(TRAINER_CFG.seed)
    ensure_fresh_experiment_name(args)
    print_run_summary(runtime_devices, args.num_nodes, run_profile)

    if args.stat_only:
        run_stat_only()
        return

    if args.mode in ("predict", "export"):
        data_config = infer_data_config(DATA_CFG.data_config_path)
    else:
        dm = WeaverDataModule(DATA_CFG, runtime_devices, args.num_nodes)
        data_config = dm.data_config

    model_cfg = copy.deepcopy(MODEL_CFG)
    model_cfg.input_dim = len(data_config.input_dicts["pf_features"])
    model_cfg.for_inference = args.mode == "export"
    model_cfg.use_amp = should_enable_model_amp(
        args.mode,
        cuda_available=torch.cuda.is_available(),
        precision=TRAINER_CFG.precision,
    )
    lit_model = ParticleTransformerHH4bLightningModule(
        model_cfg,
        OPTIM_CFG,
        SCHED_CFG,
    )
    num_params = sum(param.numel() for param in lit_model.model.parameters())
    num_trainable_params = sum(param.numel() for param in lit_model.model.parameters() if param.requires_grad)

    if args.mode == "predict":
        ckpt_path = args.ckpt_path or default_checkpoint()
        if ckpt_path is None:
            raise FileNotFoundError("No checkpoint provided and no default last.ckpt found under the output directory.")
        predict_output = Path(args.predict_output) if args.predict_output else default_predict_output()
        run_prediction(lit_model, ckpt_path, predict_output, runtime_devices, args.smoke_test)
        return

    if args.mode == "export":
        ckpt_path = args.ckpt_path or default_checkpoint()
        if ckpt_path is None:
            raise FileNotFoundError("No checkpoint provided and no default last.ckpt found under the output directory.")
        export_output = Path(args.export_output) if args.export_output else default_export_output()
        export_onnx(lit_model, ckpt_path, export_output, data_config)
        return

    trainer = build_trainer(args, runtime_devices, run_profile["num_epochs"], dm)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_start_fit = time.time()
    trainer.fit(lit_model, datamodule=dm, ckpt_path=args.ckpt_path)
    if not trainer.is_global_zero:
        return
    training_seconds = time.time() - t_start_fit
    peak_vram_mb = (torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0
    checkpoint_path = normalize_checkpoint_path(trainer.checkpoint_callback.best_model_path or default_checkpoint())
    best_tracker = get_best_val_tracker(trainer)
    best_metrics = best_tracker.best_metrics if best_tracker is not None else {}
    summary = {
        "best_val_loss": best_metrics.get("best_val_loss"),
        "best_val_acc": best_metrics.get("best_val_acc"),
        "best_epoch": best_metrics.get("best_epoch"),
        "best_global_step": best_metrics.get("best_global_step"),
        "training_seconds": round(training_seconds, 1),
        "total_seconds": round(time.time() - t_start_total, 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "num_params_M": round(num_params / 1e6, 3),
        "num_trainable_params_M": round(num_trainable_params / 1e6, 3),
        "train_steps": int(trainer.global_step),
        "train_steps_per_epoch": int(dm.train_steps_per_epoch),
        "val_steps_per_epoch": int(dm.val_steps_per_epoch),
        "checkpoint_path": checkpoint_path,
        "experiment_name": EXPERIMENT_NAME,
        "run_profile": run_profile["run_profile"],
        "output_dir": str(get_output_dir()),
        "tensorboard_dir": str(get_tensorboard_dir()),
    }
    write_train_summary(summary)
    print_train_summary(summary)


if __name__ == "__main__":
    main()
