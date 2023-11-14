import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize

from src.utils.common import pad_if_needed


###################
# Load Functions
###################


def load_chunk_features_stride(
    duration: int,
    feature_names: list[str],
    series_ids: Optional[list[str]],
    processed_dir: Path,
    phase: str,
    stride: int = 2880,  # 4h=2880step
) -> dict[str, np.ndarray]:
    features = {}

    if series_ids is None:
        series_ids = [series_dir.name for series_dir in (processed_dir / phase).glob("*")]

    for series_id in series_ids:
        series_dir = processed_dir / phase / series_id
        this_feature = []
        for feature_name in feature_names:
            this_feature.append(np.load(series_dir / f"{feature_name}.npy"))
        this_feature = np.stack(this_feature, axis=1)
        start = 0
        end = duration
        while start < len(this_feature):
            chunk_feature = this_feature[start:end]
            chunk_feature = pad_if_needed(chunk_feature, duration, pad_value=0)
            features[f"{series_id}_{start}_{end}"] = chunk_feature
            start += stride
            end += stride
    return features


def load_chunk_features(
    duration: int,
    feature_names: list[str],
    series_ids: Optional[list[str]],
    processed_dir: Path,
    phase: str,
    stride: int = 0,  # 初期値をstride だけずらしてchunkにする
    debug: bool = False,
) -> dict[str, np.ndarray]:
    features = {}

    if series_ids is None:
        series_ids = [series_dir.name for series_dir in (processed_dir / phase).glob("*")]

    for series_id in series_ids:
        series_dir = processed_dir / phase / series_id
        this_feature = []
        for feature_name in feature_names:
            this_feature.append(np.load(series_dir / f"{feature_name}.npy"))
        this_feature = np.stack(this_feature, axis=1)
        num_chunks = (len(this_feature) // duration) + 1
        this_feature = pad_if_needed(this_feature, stride + num_chunks * duration, pad_value=0)
        for i in range(num_chunks):
            chunk_feature = this_feature[stride + i * duration : stride + (i + 1) * duration]
            # chunk_feature = pad_if_needed(chunk_feature, duration, pad_value=0)
            features[f"{series_id}_{i:07}"] = chunk_feature
            if debug:
                break

    return features  # type: ignore


###################
# Augmentation
###################
def random_crop(pos: int, duration: int, max_end) -> tuple[int, int]:
    """Randomly crops with duration length including pos.
    However, 0<=start, end<=max_end
    """
    start = random.randint(max(0, pos - duration), min(pos, max_end - duration))
    end = start + duration
    return start, end


###################
# Label
###################
def get_label(this_event_df: pd.DataFrame, num_frames: int, duration: int, start: int, end: int) -> np.ndarray:
    """
    (num_frames,3) のラベルを作成
    duration は step 単位であり、num_frames は 出力につかうframe数を表しているためアップサンプリングやダウンサンプリングの影響で変化するケースがあるので注意。
    """

    # # (start, end)の範囲と(onset, wakeup)の範囲が重なるものを取得
    # ざっくりと当てはまりそうな範囲でフィルタリング
    this_event_df = this_event_df.query("@start <= wakeup & onset <= @end")

    label = np.zeros((num_frames, 3))
    # onset, wakeup, sleepのラベルを作成
    for onset, wakeup in this_event_df[["onset", "wakeup"]].to_numpy():
        onset = int((onset - start) / duration * num_frames)
        wakeup = int((wakeup - start) / duration * num_frames)
        if onset >= 0 and onset < num_frames:
            label[onset, 1] = 1
        if wakeup < num_frames and wakeup >= 0:
            label[wakeup, 2] = 1

        onset = max(0, onset)
        wakeup = min(num_frames, wakeup)
        label[onset:wakeup, 0] = 1  # sleep

    return label


# ref: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/360236#2004730
def gaussian_kernel(length: int, sigma: int = 3) -> np.ndarray:
    x = np.ogrid[-length : length + 1]
    h = np.exp(-(x**2) / (2 * sigma * sigma))  # type: ignore
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_label(label: np.ndarray, offset: int, sigma: int) -> np.ndarray:
    num_events = label.shape[1]
    for i in range(num_events):
        label[:, i] = np.convolve(label[:, i], gaussian_kernel(offset, sigma), mode="same")

    return label


def negative_sampling(this_event_df: pd.DataFrame, num_steps: int) -> int:
    """negative sampling

    Args:
        this_event_df (pd.DataFrame): event df
        num_steps (int): number of steps in this series

    Returns:
        int: negative sample position
    """
    # onsetとwakupを除いた範囲からランダムにサンプリング
    positive_positions = set(this_event_df[["onset", "wakeup"]].to_numpy().flatten().tolist())
    negative_positions = list(set(range(num_steps)) - positive_positions)
    return random.sample(negative_positions, 1)[0]


###################
# Dataset
###################
def nearest_valid_size(input_size: int, downsample_rate: int) -> int:
    """
    (x // hop_length) % 32 == 0
    を満たすinput_sizeに最も近いxを返す
    """

    while (input_size // downsample_rate) % 32 != 0:
        input_size += 1
    assert (input_size // downsample_rate) % 32 == 0

    return input_size


class TrainDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        chunk_features: dict[str, np.ndarray],
        event_df: pl.DataFrame,
        use_cache: bool = True,
    ):
        self.cfg = cfg
        self.chunk_features = chunk_features
        self.keys = list(chunk_features.keys())
        self.event_df: pd.DataFrame = (
            event_df.pivot(index=["series_id", "night"], columns="event", values="step").drop_nulls().to_pandas()
        )  # columns: onset, wakeup
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )
        self.use_cache = use_cache
        self.cache = dict((idx, None) for idx in range(len(self)))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.use_cache and self.cache[idx] is not None:
            return self.cache[idx]
        else:
            key = self.keys[idx]
            feature = self.chunk_features[key]
            feature = torch.FloatTensor(feature.T).unsqueeze(0)  # (1, num_features, duration)
            feature = resize(
                feature,
                size=[self.num_features, self.upsampled_num_frames],
                antialias=False,
            ).squeeze(0)

            series_id, start, end = key.split("_")
            start, end = int(start), int(end)

            # from hard label to gaussian label
            num_frames = self.upsampled_num_frames // self.cfg.downsample_rate
            this_event_df = self.event_df.query("series_id == @series_id").reset_index(drop=True)
            label = get_label(this_event_df, num_frames, self.cfg.duration, start, end)
            label[:, [1, 2]] = gaussian_label(
                label[:, [1, 2]], offset=self.cfg.offset, sigma=self.cfg.sigma
            )  # onset, wakeup のみハードラベルなのでガウシアンラベルに変換

            self.cache[idx] = {
                "series_id": series_id,
                "feature": feature,  # (num_features, upsampled_num_frames)
                "label": torch.FloatTensor(label),  # (pred_length, num_classes)
            }
            return self.cache[idx]


class ValidDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        chunk_features: dict[str, np.ndarray],
        event_df: pl.DataFrame,
    ):
        self.cfg = cfg
        self.chunk_features = chunk_features
        self.keys = list(chunk_features.keys())
        self.event_df = (
            event_df.pivot(index=["series_id", "night"], columns="event", values="step").drop_nulls().to_pandas()
        )
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = self.chunk_features[key]
        feature = torch.FloatTensor(feature.T).unsqueeze(0)  # (1, num_features, duration)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        series_id, chunk_id = key.split("_")
        chunk_id = int(chunk_id)
        start = chunk_id * self.cfg.duration
        end = start + self.cfg.duration
        num_frames = self.upsampled_num_frames // self.cfg.downsample_rate
        label = get_label(
            self.event_df.query("series_id == @series_id").reset_index(drop=True),
            num_frames,
            self.cfg.duration,
            start,
            end,
        )
        return {
            "key": key,
            "feature": feature,  # (num_features, duration)
            "label": torch.FloatTensor(label),  # (duration, num_classes)
        }


class TestDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        chunk_features: dict[str, np.ndarray],
    ):
        self.cfg = cfg
        self.chunk_features = chunk_features
        self.keys = list(chunk_features.keys())
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = self.chunk_features[key]
        feature = torch.FloatTensor(feature.T).unsqueeze(0)  # (1, num_features, duration)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        return {
            "key": key,
            "feature": feature,  # (num_features, duration)
        }


###################
# DataModule
###################
class SegDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig, fold: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(cfg.dir.data_dir)
        self.processed_dir = Path(cfg.dir.processed_dir)
        self.event_df = pl.read_csv(self.data_dir / "train_events.csv").drop_nulls()
        self.fold = fold

        if self.fold is None:  # single fold
            self.train_series_ids = self.cfg.split.train_series_ids
            self.valid_series_ids = self.cfg.split.valid_series_ids
        else:
            self.train_series_ids = self.cfg[f"fold_{fold}"].train_series_ids
            self.valid_series_ids = self.cfg[f"fold_{fold}"].valid_series_ids

        self.train_event_df = self.event_df.filter(pl.col("series_id").is_in(self.train_series_ids)).filter(
            ~pl.col("series_id").is_in(self.cfg.ignore.train)
        )
        self.valid_event_df = self.event_df.filter(pl.col("series_id").is_in(self.valid_series_ids))

        # train data
        self.train_chunk_features = load_chunk_features_stride(
            duration=self.cfg.duration,
            feature_names=self.cfg.features,
            series_ids=self.train_series_ids,
            processed_dir=self.processed_dir,
            phase="train",
            stride=cfg.datamodule.train_stride,
        )

        # valid data
        self.valid_chunk_features = load_chunk_features(
            duration=self.cfg.duration,
            feature_names=self.cfg.features,
            series_ids=self.valid_series_ids,
            processed_dir=self.processed_dir,
            phase="train",
        )

    def train_dataloader(self):
        train_dataset = TrainDataset(
            cfg=self.cfg,
            chunk_features=self.train_chunk_features,
            event_df=self.train_event_df,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_dataset = ValidDataset(
            cfg=self.cfg,
            chunk_features=self.valid_chunk_features,
            event_df=self.valid_event_df,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        return valid_loader
