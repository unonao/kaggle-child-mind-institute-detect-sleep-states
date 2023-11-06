import logging
from pathlib import Path
import numpy as np
import polars as pl
import os
import hydra
from omegaconf import DictConfig
from src.datamodule.seg import SegDataModule


from src.utils.score import score_ternary_search_th, score_ternary_search_distance

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")
LOGGER = logging.getLogger(Path(__file__).name)


@hydra.main(config_path="conf", config_name="search", version_base="1.2")
def main(cfg: DictConfig):  # type: ignore
    """
    以下をロードする
            np.save("keys.npy", np.array(keys))
            np.save("labels.npy", labels)
            np.save("preds.npy", preds)
            val_pred_df.write_csv("val_pred_df.csv")
    """
    # print(cfg)

    event_df = pl.read_csv(Path(cfg.train.dir.data_dir) / "train_events.csv").drop_nulls()
    val_pred_df = event_df.filter(pl.col("series_id").is_in(cfg.train.split.valid_series_ids))
    distance = cfg.train.post_process.distance

    #
    exp_dir = Path(os.path.join(cfg.base_dir, cfg.exp, "single"))

    keys = np.load(exp_dir / "keys.npy")
    labels = np.load(exp_dir / "labels.npy")
    preds = np.load(exp_dir / "preds.npy")

    # search th
    """
    score, th = score_ternary_search_th(
        val_event_df=val_pred_df,
        keys=keys,
        preds=preds,
        distance=distance,
    )
    """

    # search distance
    score, th = score_ternary_search_distance(
        val_event_df=val_pred_df,
        keys=keys,
        preds=preds,
        score_th=cfg.train.post_process.score_th,
    )

    LOGGER.info(f"score: {score}, th: {th}")

    return


if __name__ == "__main__":
    main()
