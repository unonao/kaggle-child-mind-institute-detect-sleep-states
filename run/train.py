import logging
from pathlib import Path
import wandb

import hydra
from hydra.core.hydra_config import HydraConfig
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

from src.datamodule.seg import SegDataModule
from src.datamodule.seg_stride import SegDataModule as SegDataModuleStride
from src.datamodule.seg_overlap import SegDataModule as SegDataModuleOverlap
from src.modelmodule.seg import SegModel
from src.utils.metrics import event_detection_ap


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    job_name = ""
    print(HydraConfig.get().job)
    if "num" in HydraConfig.get().job:
        job_name = str(HydraConfig.get().job.num)
        print(job_name)
    else:
        print("No job name")

    pl_logger = WandbLogger(
        name=cfg.exp_name + "_" + job_name,
        project="child-mind-institute-detect-sleep-states-single",
        reinit=True,
    )
    pl_logger.log_hyperparams(cfg)

    # init lightning model
    if cfg.datamodule.how == "random":
        datamodule = SegDataModule(cfg)
    elif cfg.datamodule.how == "stride":
        datamodule = SegDataModuleStride(cfg)
    elif cfg.datamodule.how == "overlap":
        datamodule = SegDataModuleOverlap(cfg)

    model = SegModel(
        cfg,
        datamodule.valid_event_df,
        len(cfg.features),
        len(cfg.labels),
        cfg.duration,
        datamodule,
    )

    # set callbacks
    checkpoint_cb = ModelCheckpoint(
        verbose=True,
        monitor=cfg.monitor,
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
        sync_batchnorm=True,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
    )

    trainer.fit(model, datamodule=datamodule)
    wandb.finish()
    return


if __name__ == "__main__":
    main()
