import numpy as np
import polars as pl
from scipy.signal import find_peaks
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")
LOGGER = logging.getLogger(Path(__file__).name)


def post_process_for_seg_group_by_day(keys: list[str], preds: np.ndarray, val_df: pl.DataFrame) -> pl.DataFrame:
    """
    Args:
        keys (list[str]): 予測したchunkのkey({series_id}_{chunk_id})
        preds (np.ndarray): (chunk_num, duration, 2)
        val_df (pl.DataFrame): sequence
    """
    # valid の各ステップに予測結果を付与
    count_df = val_df.get_column("series_id").value_counts()
    series2numsteps_dict = dict(count_df.select("series_id", "counts").iter_rows())

    # 順序を保ったままseries_idを取得
    unique_series_ids = val_df.get_column("series_id").unique(maintain_order=True).to_list()
    key_series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))

    # val_dfに合わせた順番でpredsから予測結果を取得
    preds_list = []
    for series_id in unique_series_ids:
        series_idx = np.where(key_series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 2)
        this_series_preds = this_series_preds[: series2numsteps_dict[series_id], :]
        preds_list.append(this_series_preds)

    preds_all = np.concatenate(preds_list, axis=0)
    valid_preds_df = val_df.with_columns(
        pl.Series(name="prediction_onset", values=preds_all[:, 0]),
        pl.Series(name="prediction_wakeup", values=preds_all[:, 1]),
    )
    valid_preds_df = valid_preds_df.with_columns(pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"))

    # from sakami-san code
    def make_submission(preds_df: pl.DataFrame) -> pl.DataFrame:
        event_dfs = [
            preds_df.with_columns(pl.lit(event).alias("event"), pl.col("timestamp").dt.date().alias("date"))
            .group_by(["series_id", "date"])
            .agg(pl.all().sort_by(f"prediction_{event}").last())
            .rename({f"prediction_{event}": "score"})
            .select(["series_id", "step", "event", "score"])
            for event in ["onset", "wakeup"]
        ]
        submission_df = (
            pl.concat(event_dfs)
            .sort(["series_id", "step"])
            .with_columns(pl.arange(0, pl.count()).alias("row_id"))
            .select(["row_id", "series_id", "step", "event", "score"])
        )
        return submission_df

    submission_df = make_submission(valid_preds_df)

    return submission_df


def post_process_for_seg(
    keys: list[str],
    preds: np.ndarray,
    score_th: float = 0.01,
    distance: int = 5000,
    periodicity_dict: dict[np.ndarray] | None = None,
) -> pl.DataFrame:
    """make submission dataframe for segmentation task

    Args:
        keys (list[str]): list of keys. key is "{series_id}_{chunk_id}"
        preds (np.ndarray): (num_series * num_chunks, duration, 2)
        score_th (float, optional): threshold for score. Defaults to 0.5.
        distance (int, optional): distance for peaks. Defaults to 5000.
        periodicity_dict (dict[np.ndarray], optional): series_id を key に periodicity の 1d の予測結果を持つ辞書. 値は 0 or 1 の np.ndarray. Defaults to None.

    Returns:
        pl.DataFrame: submission dataframe
    """
    LOGGER.info("is periodicity_dict None? : {}".format(periodicity_dict is None))

    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    unique_series_ids = np.unique(series_ids)

    records = []
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 2)
        if periodicity_dict is not None:
            this_series_preds = this_series_preds[: len(periodicity_dict[series_id]), :]
            this_series_preds[periodicity_dict[series_id] > 0.5] = 0  # periodicity があるところは0にする

        for i, event_name in enumerate(["onset", "wakeup"]):
            this_event_preds = this_series_preds[:, i]
            steps = find_peaks(this_event_preds, height=score_th, distance=distance)[0]
            scores = this_event_preds[steps]

            for step, score in zip(steps, scores):
                records.append(
                    {
                        "series_id": series_id,
                        "step": step,
                        "event": event_name,
                        "score": score,
                    }
                )

    if len(records) == 0:  # 一つも予測がない場合はdummyを入れる
        records.append(
            {
                "series_id": series_id,
                "step": 0,
                "event": "onset",
                "score": 0,
            }
        )

    sub_df = pl.DataFrame(records).sort(by=["series_id", "step"])
    row_ids = pl.Series(name="row_id", values=np.arange(len(sub_df)))
    sub_df = sub_df.with_columns(row_ids).select(["row_id", "series_id", "step", "event", "score"])
    return sub_df
