from typing import Optional

import numpy as np
import polars as pl
import torch
import torch.optim as optim
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torchvision.transforms.functional import resize
from transformers import get_cosine_schedule_with_warmup
import torch.nn as nn
import math

from src.datamodule.seg import nearest_valid_size
from src.models.common import get_model
from src.utils.metrics import event_detection_ap
from src.utils.post_process import post_process_for_seg
from src.utils.periodicity import get_periodicity_dict


class SegModel(LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        val_event_df: pl.DataFrame,
        feature_dim: int,
        num_classes: int,
        duration: int,
        datamodule=None,
        fold: int | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.val_event_df = val_event_df
        self.duration = duration
        valid_duration = nearest_valid_size(int(duration * cfg.upsample_rate), cfg.downsample_rate)
        self.model = get_model(
            cfg,
            feature_dim=feature_dim,
            n_classes=num_classes,
            num_timesteps=valid_duration // cfg.downsample_rate,
        )
        self.postfix = f"_fold{fold}" if fold is not None else ""
        self.validation_step_outputs: list = []
        self.__best_loss = np.inf
        self.__best_score = 0.0
        self.datamodule = datamodule
        self.epoch = 0

        self.overlap = cfg.datamodule.overlap

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> dict[str, Optional[torch.Tensor]]:
        return self.model(x, labels)

    def training_step(self, batch, batch_idx):
        return self.__share_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.__share_step(batch, "val")

    def __share_step(self, batch, mode: str) -> torch.Tensor:
        if mode == "train":
            do_mixup = np.random.rand() < self.cfg.augmentation.mixup_prob
            do_cutmix = np.random.rand() < self.cfg.augmentation.cutmix_prob
        elif mode == "val":
            do_mixup = False
            do_cutmix = False

        output = self.model(batch["feature"], batch["label"], batch["masks"], do_mixup, do_cutmix)
        loss: torch.Tensor = output["loss"]
        logits = output["logits"]  # (batch_size, n_timesteps, n_classes)

        if mode == "train":
            self.log(
                f"{mode}_loss{self.postfix}",
                loss.detach().item(),
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )
        elif mode == "val":
            # アップサンプリングやダウンサンプリングで長さが変わっているのでリサイズしてもとに戻す
            resized_logits = resize(
                logits.sigmoid().detach().cpu(),
                size=[self.duration, logits.shape[2]],
                antialias=False,
            )
            resized_labels = resize(
                batch["label"].detach().cpu(),
                size=[self.duration, logits.shape[2]],
                antialias=False,
            )
            self.validation_step_outputs.append(
                (
                    batch["key"],
                    resized_labels.numpy(),
                    resized_logits.numpy(),
                    loss.detach().item(),
                )
            )
            self.log(
                f"{mode}_loss{self.postfix}",
                loss.detach().item(),
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )

        return loss

    def on_train_epoch_end(self):
        if (self.cfg.sigma_decay is not None) and (self.datamodule is not None):
            self.datamodule.set_sigma(self.datamodule.sigma * self.cfg.sigma_decay)
        if self.cfg.sleep_decay is not None:
            self.model.update_loss_fn(self.cfg.sleep_decay)
        self.epoch += 1
        self.datamodule.set_now_epoch(self.epoch)

    def on_validation_epoch_end(self):
        keys = []
        for x in self.validation_step_outputs:
            keys.extend(x[0])
        l = self.overlap if self.overlap > 0 else None
        r = -self.overlap if self.overlap > 0 else None
        labels = np.concatenate([x[1][:, l:r, :] for x in self.validation_step_outputs])
        preds = np.concatenate([x[2][:, l:r, :] for x in self.validation_step_outputs])
        losses = np.array([x[3] for x in self.validation_step_outputs])
        loss = losses.mean()

        print(preds.shape)

        periodicity_dict = get_periodicity_dict(self.cfg)
        val_pred_df = post_process_for_seg(
            keys=keys,
            preds=preds[:, :, [1, 2]],
            score_th=self.cfg.post_process.score_th,
            distance=self.cfg.post_process.distance,
            periodicity_dict=periodicity_dict,
        )
        print(self.val_event_df.head())
        print(val_pred_df.head())
        score = event_detection_ap(self.val_event_df.to_pandas(), val_pred_df.to_pandas())
        self.log(f"val_score{self.postfix}", score, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        if ((self.cfg.monitor == "val_score") and (score > self.__best_score)) or (
            (self.cfg.monitor == "val_loss") and (loss < self.__best_loss)
        ):
            np.save(f"keys{self.postfix}.npy", np.array(keys))
            np.save(f"labels{self.postfix}.npy", labels)
            np.save(f"preds{self.postfix}.npy", preds)
            val_pred_df.write_csv(f"val_pred_df{self.postfix}.csv")
            torch.save(self.model.state_dict(), f"best_model{self.postfix}.pth")
            print(f"Saved best model {self.__best_score} -> {score}")
            print(f"Saved best model {self.__best_loss} -> {loss}")
            self.__best_score = score
            self.__best_loss = loss

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr)

        # 1epoch分をwarmupとするための記述
        num_warmup_steps = (
            math.ceil(self.trainer.max_steps / self.cfg.epoch) * 1 if self.cfg.scheduler.use_warmup else 0
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=self.trainer.max_steps, num_warmup_steps=num_warmup_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
