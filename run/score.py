import logging
import os
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig

from src.utils.common import trace
from src.utils.metrics import event_detection_ap
from src.utils.periodicity import get_periodicity_dict
from src.utils.post_process import post_process_for_seg
from src.utils.score import score_group_by_day, score_ternary_search_distance, score_ternary_search_th

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")
LOGGER = logging.getLogger(Path(__file__).name)


@hydra.main(config_path="conf", config_name="score", version_base="1.2")
def main(cfg: DictConfig):  # type: ignore
    event_df = pl.read_csv(Path(cfg.dir.data_dir) / "train_events.csv").drop_nulls()
    valid_event_df = event_df.filter(pl.col("series_id").is_in(cfg.split.valid_series_ids))
    distance = cfg.post_process.distance

    # 予測結果の読み込み
    exp_dir = Path(os.path.join(cfg.base_dir, cfg.exp, "single"))
    keys = np.load(exp_dir / "keys.npy")
    # labels = np.load(exp_dir / "labels.npy")
    preds = np.load(exp_dir / "preds.npy")

    # search th
    if cfg.how == "score":
        periodicity_dict = None
        if cfg.score.remove_periodicity:
            with trace("get periodicity_dict"):
                periodicity_dict = get_periodicity_dict(cfg)
        submission_df = post_process_for_seg(
            keys,
            preds[:, :, [1, 2]],
            score_th=cfg.post_process.score_th,
            distance=cfg.post_process.distance,
            periodicity_dict=periodicity_dict,
        )
        score = event_detection_ap(
            valid_event_df.to_pandas(),
            submission_df.to_pandas(),
        )
        LOGGER.info(f"score: {score:.4f}")

    elif cfg.how == "threshold":
        score, th = score_ternary_search_th(
            val_event_df=valid_event_df,
            keys=keys,
            preds=preds[:, :, [1, 2]],
            distance=distance,
        )
        LOGGER.info(f"score: {score:.4f}, th: {th:.4f}")
    elif cfg.how == "distance":
        # search distance
        score, distance = score_ternary_search_distance(
            val_event_df=valid_event_df,
            keys=keys,
            preds=preds[:, :, [1, 2]],
            score_th=cfg.post_process.score_th,
        )
        LOGGER.info(f"score: {score:.4f}, distance: {distance:.4f}")
    elif cfg.how == "group_by_day":
        train_df = pl.read_parquet(Path(cfg.dir.data_dir) / "train_series.parquet")
        valid_df = train_df.filter(pl.col("series_id").is_in(cfg.split.valid_series_ids))
        score = score_group_by_day(
            val_event_df=valid_event_df,
            keys=keys,
            preds=preds[:, :, [1, 2]],
            val_df=valid_df,
        )
        LOGGER.info(f"score: {score:.4f}")

    return


if __name__ == "__main__":
    main()
