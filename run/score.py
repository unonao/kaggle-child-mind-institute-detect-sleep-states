import logging
from pathlib import Path
import numpy as np
import polars as pl
import os
import hydra
from omegaconf import DictConfig
from src.datamodule.seg import SegDataModule


from src.utils.score import score_ternary_search_th, score_ternary_search_distance, score_group_by_day

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")
LOGGER = logging.getLogger(Path(__file__).name)


@hydra.main(config_path="conf", config_name="score", version_base="1.2")
def main(cfg: DictConfig):  # type: ignore
    train_df = pl.read_parquet(Path(cfg.dir.data_dir) / "train_series.parquet")
    valid_df = train_df.filter(pl.col("series_id").is_in(cfg.split.valid_series_ids))

    event_df = pl.read_csv(Path(cfg.dir.data_dir) / "train_events.csv").drop_nulls()
    valid_event_df = event_df.filter(pl.col("series_id").is_in(cfg.split.valid_series_ids))
    distance = cfg.post_process.distance

    # 予測結果の読み込み
    exp_dir = Path(os.path.join(cfg.base_dir, cfg.exp, "single"))
    keys = np.load(exp_dir / "keys.npy")
    # labels = np.load(exp_dir / "labels.npy")
    preds = np.load(exp_dir / "preds.npy")

    # search th
    if cfg.how == "threshold":
        score, th = score_ternary_search_th(
            val_event_df=valid_event_df,
            keys=keys,
            preds=preds[:, :, [1, 2]],
            distance=distance,
        )
        LOGGER.info(f"score: {score}, th: {th}")
    elif cfg.how == "distance":
        # search distance
        score, distance = score_ternary_search_distance(
            val_event_df=valid_event_df,
            keys=keys,
            preds=preds[:, :, [1, 2]],
            score_th=cfg.post_process.score_th,
        )
        LOGGER.info(f"score: {score}, distance: {distance}")
    elif cfg.how == "group_by_day":
        score = score_group_by_day(
            val_event_df=valid_event_df,
            keys=keys,
            preds=preds[:, :, [1, 2]],
            val_df=valid_df,
        )
        LOGGER.info(f"score: {score}")

    return


if __name__ == "__main__":
    main()
