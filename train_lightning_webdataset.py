import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from dataloader.webdataset_depth import WebDatasetDepthDataModule
from network import VAENet


class VAELoss(nn.Module):
    def __init__(self, scale: float = 10.0):
        super().__init__()
        self.reco_loss = nn.HuberLoss(reduction="mean")
        self.scale = scale

    def forward(self, output_dict, target, beta):
        recon = output_dict["depth"]
        mu = output_dict["mu"]
        logvar = output_dict["logvar"]

        if recon.shape[-2:] != target.shape[-2:]:
            recon = F.interpolate(recon, size=target.shape[-2:], mode="bilinear", align_corners=True)

        recon_loss = self.reco_loss(recon, target)
        kl_loss = torch.mean(-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()))
        loss = recon_loss * self.scale + beta * kl_loss
        return loss, recon_loss, kl_loss


class DepthVAELightningModule(pl.LightningModule):
    def __init__(self, model_config: dict, training_config: dict):
        super().__init__()
        self.save_hyperparameters({"model": model_config, "training": training_config})
        self.model = VAENet(
            latent_dim=model_config.get("latent_dim", 64),
            in_channels=model_config.get("in_channels", 3),
            out_channels=model_config.get("out_channels", 1),
        )
        self.criterion = VAELoss(scale=training_config.get("loss_scale", 10.0))
        self.learning_rate = training_config.get("learning_rate", 1e-3)
        self.weight_decay = training_config.get("weight_decay", 1e-4)
        self.epochs = training_config.get("epochs", 10)
        self.init_beta = training_config.get("init_beta", 0.1)
        self.final_beta = training_config.get("final_beta", 0.5)

    def _beta(self):
        if self.epochs <= 1:
            return self.final_beta
        progress = min(float(self.current_epoch) / float(self.epochs - 1), 1.0)
        if self.init_beta <= 0:
            return self.init_beta + (self.final_beta - self.init_beta) * progress
        return self.init_beta * (self.final_beta / self.init_beta) ** progress

    def training_step(self, batch, batch_idx):
        depth_input, depth_target = batch
        output = self.model(depth_input)
        beta = self._beta()
        loss, recon_loss, kl_loss = self.criterion(output, depth_target, beta)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/recon_loss", recon_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/kl_loss", kl_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/beta", beta, on_step=True, on_epoch=False, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        depth_input, depth_target = batch
        output = self.model(depth_input)
        beta = self._beta()
        loss, recon_loss, kl_loss = self.criterion(output, depth_target, beta)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/recon_loss", recon_loss, on_epoch=True, sync_dist=True)
        self.log("val/kl_loss", kl_loss, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=self.hparams["training"].get("eta_min", 1e-6),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(config_path):
    config = load_config(config_path)
    dataloader_config = config.get("dataloader", {})
    training_config = config.get("training", {})
    model_config = config.get("model", {})

    output_root = training_config.get("output_path", "output")
    run_name = training_config.get("run_name", "webdataset-pretrain")
    output_dir = os.path.join(output_root, run_name)
    ckpt_dir = os.path.join(output_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = False
    if training_config.get("is_wandb", False):
        logger = WandbLogger(
            project=training_config.get("wandb_project", "reconstruction-vae"),
            entity=training_config.get("wandb_entity"),
            name=f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            save_dir=output_dir,
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=training_config.get("save_top_k", 3),
        save_last=True,
    )

    datamodule = WebDatasetDepthDataModule(dataloader_config, training_config)
    model = DepthVAELightningModule(model_config, training_config)

    last_ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
    resume_ckpt = last_ckpt_path if os.path.isfile(last_ckpt_path) else None
    num_gpus = training_config.get("num_gpus", 1)
    accelerator = "gpu" if torch.cuda.is_available() and num_gpus != 0 else "cpu"
    devices = num_gpus if accelerator == "gpu" else "auto"
    strategy = DDPStrategy(find_unused_parameters=False) if accelerator == "gpu" and num_gpus != 1 else "auto"

    callbacks = [checkpoint_callback]
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    precision = training_config.get("precision", "16-mixed" if accelerator == "gpu" else "32-true")
    if accelerator == "cpu" and precision == "16-mixed":
        precision = "32-true"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=training_config.get("epochs", 10),
        max_steps=training_config.get("max_steps", -1),
        precision=precision,
        gradient_clip_val=training_config.get("gradient_clip_val", 1.0),
        callbacks=callbacks,
        logger=logger,
        default_root_dir=output_dir,
        limit_train_batches=training_config.get("limit_train_batches", 1.0),
        limit_val_batches=training_config.get("limit_val_batches", 1.0),
        log_every_n_steps=training_config.get("log_every_n_steps", 50),
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU Lightning VAE pretraining from WebDataset depth PNG shards")
    parser.add_argument("--config", type=str, default="config/pretrain_webdataset.yaml")
    args = parser.parse_args()
    main(args.config)
