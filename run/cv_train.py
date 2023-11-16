import logging
from pathlib import Path
import polars as pl

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
import wandb
import gc

from src.datamodule.seg import SegDataModule
from src.datamodule.seg_stride import SegDataModule as SegDataModuleStride
from src.modelmodule.seg import SegModel
from src.utils.metrics import event_detection_ap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")
LOGGER = logging.getLogger(Path(__file__).name)


@hydra.main(config_path="conf", config_name="cv_train", version_base="1.2")
def main(cfg: DictConfig):  # type: ignore
    seed_everything(cfg.seed)

    # init experiment logger
    pl_logger = WandbLogger(
        name=cfg.exp_name,
        project="child-mind-institute-detect-sleep-states",
    )
    for fold in range(cfg.num_fold):
        LOGGER.info(f"Start Training Fold {fold}")
        # init lightning model
        if cfg.datamodule.how == "random":
            datamodule = SegDataModule(cfg, fold)
        elif cfg.datamodule.how == "stride":
            datamodule = SegDataModuleStride(cfg, fold)
        LOGGER.info("Set Up DataModule")
        model = SegModel(
            cfg, datamodule.valid_event_df, len(cfg.features), len(cfg.labels), cfg.duration, datamodule, fold
        )

        # set callbacks
        checkpoint_cb = ModelCheckpoint(
            verbose=True,
            monitor=f"{cfg.monitor}_fold{fold}",
            mode=cfg.monitor_mode,
            save_top_k=1,
            save_last=False,
        )
        lr_monitor = LearningRateMonitor("epoch")
        progress_bar = RichProgressBar()
        model_summary = RichModelSummary(max_depth=2)

        trainer = Trainer(
            # env
            default_root_dir=Path.cwd(),
            # num_nodes=cfg.training.num_gpus,
            accelerator=cfg.accelerator,
            precision=16 if cfg.use_amp else 32,
            # training
            fast_dev_run=cfg.debug,  # run only 1 train batch and 1 val batch
            max_epochs=cfg.epoch,
            max_steps=cfg.epoch * len(datamodule.train_dataloader()),
            gradient_clip_val=cfg.gradient_clip_val,
            accumulate_grad_batches=cfg.accumulate_grad_batches,
            callbacks=[checkpoint_cb, lr_monitor, progress_bar, model_summary],
            logger=pl_logger,
            # resume_from_checkpoint=resume_from,
            num_sanity_val_steps=0,
            log_every_n_steps=int(len(datamodule.train_dataloader()) * 0.1),
            reload_dataloaders_every_n_epochs=1,
            sync_batchnorm=True,
            check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        )

        trainer.fit(model, datamodule=datamodule)

        # load best weights
        model = model.load_from_checkpoint(
            checkpoint_cb.best_model_path,
            cfg=cfg,
            val_event_df=datamodule.valid_event_df,
            feature_dim=len(cfg.features),
            num_classes=len(cfg.labels),
            duration=cfg.duration,
        )
        weights_path = str(f"model_weights_fold{fold}.pth")
        LOGGER.info(f"Extracting and saving best weights: {weights_path}")
        torch.save(model.model.state_dict(), weights_path)

        del model
        del trainer
        del datamodule
        gc.collect()

    # oof での評価
    LOGGER.info("Start OOF scoring")

    # 正解ラベルの読み込み
    train_event_df = pl.read_csv(Path(cfg.dir.data_dir) / "train_events.csv").drop_nulls()

    # 予測結果の読み込み
    oof_event_df_list = []
    for fold in range(cfg.num_fold):
        oof_event_df_list.append(pl.read_csv(f"val_pred_df_fold{fold}.csv"))
    oof_event_df = pl.concat(oof_event_df_list)

    # 評価
    LOGGER.info("Start event_detection_ap")
    score, ap_table = event_detection_ap(
        train_event_df.to_pandas(),
        oof_event_df.to_pandas(),
        with_table=True,
    )

    wandb.log({"ap_table": wandb.Table(dataframe=ap_table.reset_index()[["event", "tolerance", "ap"]])})

    for event in ["onset", "wakeup"]:
        plt.figure(figsize=(10, 10))
        for (event_key, tolerance), group in ap_table[ap_table.index.get_level_values("event") == event].iterrows():
            plt.plot(
                group["recall"][:-1], group["precision"][:-1], label=f'Tolerance: {tolerance}, AP:{group["ap"]:.3f}'
            )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curves for {event}")
        plt.legend()
        # 図をwandbにログとして保存
        wandb.log({f"pr_curve_{event}": wandb.Image(plt)})
        plt.close()

    LOGGER.info(f"OOF score: {score}")
    wandb.log({"cv_score": score})

    return


if __name__ == "__main__":
    main()
