import logging
from pathlib import Path
import polars as pl
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from src.utils.metrics import event_detection_ap
from src.utils.chokudai_search import chokudai_search_from_2nd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")
LOGGER = logging.getLogger(Path(__file__).name)


@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg: DictConfig):  # type: ignore
    LOGGER.info(OmegaConf.to_container(cfg, resolve=True))
    LOGGER.info("Start Chokudai Search")

    event_df = pl.read_csv(Path(cfg.dir.data_dir) / "train_events.csv").drop_nulls()
    event_df = event_df.with_columns(pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"))

    pred_df = (
        pl.read_parquet(cfg.path.pred_onset)
        .rename({"label_pred": "stacking_prediction_onset"})
        .drop("label")
        .join(
            pl.read_parquet(cfg.path.pred_wakeup).rename({"label_pred": "stacking_prediction_wakeup"}).drop("label"),
            on=["series_id", "step"],
            how="left",
        )
    )
    pred_df = pred_df.with_columns(
        ((pl.col("step") - pl.col("step").shift(1)) != 12)
        .cast(int)
        .cumsum()
        .over("series_id")
        .fill_null(0)
        .alias("chunk_id")
    ).with_columns(pl.col("step").cast(pl.UInt32))

    train_df = pl.read_parquet(Path(cfg.dir.data_dir) / "train_series.parquet")
    train_df = train_df.with_columns(pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z")).filter(
        pl.col("step") % 12 == 0
    )
    pred_df = pred_df.join(train_df, on=["series_id", "step"], how="left")

    print(pred_df.head(10))


if __name__ == "__main__":
    main()
