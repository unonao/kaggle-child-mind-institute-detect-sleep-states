from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
from tqdm import tqdm

from src.datamodule.seg import TestDataset, load_chunk_features, nearest_valid_size
from src.models.common import get_model
from src.utils.common import trace
from src.utils.post_process import post_process_for_seg, post_process_for_seg_group_by_day
from src.utils.periodicity import get_periodicity_dict


def load_model(cfg: DictConfig, fold: int) -> nn.Module:
    num_timesteps = nearest_valid_size(int(cfg.duration * cfg.upsample_rate), cfg.downsample_rate)
    model = get_model(
        cfg,
        feature_dim=len(cfg.features),
        n_classes=len(cfg.labels),
        num_timesteps=num_timesteps // cfg.downsample_rate,
    )

    # load weights
    if cfg.weight is not None:
        weight_path = (
            Path(cfg.dir.cv_model_dir) / cfg.weight["exp_name"] / cfg.weight["run_name"] / f"best_model_fold{fold}.pth"
        )
        model.load_state_dict(
            torch.load(weight_path), strict=False  #  Unexpected key(s) in state_dict: "loss_fn.pos_weight". の回避
        )
        print('load weight from "{}"'.format(weight_path))
    return model


def get_test_dataloader(cfg: DictConfig) -> DataLoader:
    """get test dataloader

    Args:
        cfg (DictConfig): config

    Returns:
        DataLoader: test dataloader
    """
    feature_dir = Path(cfg.dir.processed_dir) / cfg.phase
    series_ids = [x.name for x in feature_dir.glob("*")]
    chunk_features = load_chunk_features(
        duration=cfg.duration,
        feature_names=cfg.features,
        series_ids=series_ids,
        processed_dir=Path(cfg.dir.processed_dir),
        phase=cfg.phase,
    )
    test_dataset = TestDataset(cfg, chunk_features=chunk_features)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return test_dataloader


def get_test_series(cfg: DictConfig) -> pl.DataFrame:
    if cfg.phase == "train":
        test_df = pl.read_parquet(Path(cfg.dir.data_dir) / "train_series.parquet")
    elif cfg.phase == "test":
        test_df = pl.read_parquet(Path(cfg.dir.data_dir) / "test_series.parquet")
    return test_df


def inference(
    duration: int, loader: DataLoader, model: nn.Module, device: torch.device, use_amp
) -> tuple[list[str], np.ndarray]:
    model = model.to(device)
    model.eval()

    preds = []
    keys = []
    for batch in tqdm(loader, desc="inference"):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                x = batch["feature"].to(device)
                pred = model(x)["logits"].sigmoid()
                pred = resize(
                    pred.detach().cpu(),
                    size=[duration, pred.shape[2]],
                    antialias=False,
                )
            key = batch["key"]
            preds.append(pred.detach().cpu().numpy())
            keys.extend(key)

    preds = np.concatenate(preds)
    return keys, preds


@hydra.main(config_path="conf", config_name="cv_inference", version_base="1.2")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    with trace("load test dataloader"):
        test_dataloader = get_test_dataloader(cfg)

    # inference
    keys = None
    preds_list = []
    for fold in range(cfg.num_fold):
        with trace("load model"):
            model = load_model(cfg, fold)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with trace("inference"):
            keys, preds = inference(cfg.duration, test_dataloader, model, device, use_amp=cfg.use_amp)
            preds_list.append(preds)
    preds = np.mean(preds_list, axis=0)

    # make submission
    with trace("make submission"):
        if cfg.how_post_process == "peaks":
            periodicity_dict = None
            if cfg.post_process.remove_periodicity:
                with trace("get periodicity_dict"):
                    periodicity_dict = get_periodicity_dict(cfg)
            sub_df = post_process_for_seg(
                keys,
                preds[:, :, [1, 2]],
                score_th=cfg.post_process.score_th,
                distance=cfg.post_process.distance,
                periodicity_dict=periodicity_dict,
            )
        elif cfg.how_post_process == "group_by_day":
            test_df = get_test_series(cfg)
            sub_df = post_process_for_seg_group_by_day(
                keys,
                preds[:, :, [1, 2]],
                test_df,
            )
    np.save("keys.npy", np.array(keys))
    np.save("preds.npy", preds)
    sub_df.write_csv(Path(cfg.dir.sub_dir) / "submission.csv")


if __name__ == "__main__":
    main()
