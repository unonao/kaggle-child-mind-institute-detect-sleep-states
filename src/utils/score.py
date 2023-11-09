import numpy as np
import polars as pl
from tqdm.auto import tqdm

from src.utils.metrics import event_detection_ap
from src.utils.post_process import post_process_for_seg


def score_group_by_day(val_event_df: pl.DataFrame, keys: list[str], preds: np.ndarray, val_df: pl.DataFrame) -> float:
    """
    日毎に最大値のeventを検出し、それをsubmissionとしてスコアリングする
    """

    # valid の各ステップに予測結果を付与
    count_df = val_df.get_column("series_id").value_counts()
    series2numsteps_dict = dict(count_df.select("series_id", "counts").iter_rows())

    # 順序を保ったままseries_idを取得
    all_series_ids = val_df.get_column("series_id").to_numpy()
    _, idx = np.unique(all_series_ids, return_index=True)
    unique_series_ids = all_series_ids[np.sort(idx)]

    key_series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))

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

    score = event_detection_ap(
        val_event_df.to_pandas(),
        submission_df.to_pandas(),
    )
    return score


def score_ternary_search_distance(
    val_event_df: pl.DataFrame, keys: list[str], preds: np.ndarray, score_th: float = 0.005
) -> [float, float]:
    """
    post_process_for_seg のパラメータdistanceを ternary searchで探索する
    """
    l = 5
    r = 100

    cnt = 0
    best_score = 0.0
    best_distance = 0

    for cnt in tqdm(range(30)):
        if r - l < 1:
            break
        m1 = int(l + (r - l) / 3)
        m2 = int(r - (r - l) / 3)
        score1 = event_detection_ap(
            val_event_df.to_pandas(),
            post_process_for_seg(
                keys=keys,
                preds=preds,
                score_th=score_th,
                distance=m1,
            ).to_pandas(),
        )
        score2 = event_detection_ap(
            val_event_df.to_pandas(),
            post_process_for_seg(
                keys=keys,
                preds=preds,
                score_th=score_th,
                distance=m2,
            ).to_pandas(),
        )

        if score1 >= score2:
            r = m2
            best_score = score1
            best_distance = m1

        else:
            l = m1
            best_score = score2
            best_distance = m2

        tqdm.write(f"score1(m1): {score1:.5f}({m1:.5f}), score2(m2): {score2:.5f}({m2:.5f}), l: {l:.5f}, r: {r:.5f}")

        if abs(m2 - m1) <= 2:
            break

    return best_score, best_distance


def score_ternary_search_th(
    val_event_df: pl.DataFrame, keys: list[str], preds: np.ndarray, distance: int = 5000
) -> [float, float]:
    """
    post_process_for_seg のパラメータ score_th を ternary searchで探索する
    """
    l = 0.0
    r = 1.0

    cnt = 0
    best_score = 0.0
    best_th = 0.0

    for cnt in tqdm(range(30)):
        if r - l < 0.01:
            break
        m1 = l + (r - l) / 3
        m2 = r - (r - l) / 3
        score1 = event_detection_ap(
            val_event_df.to_pandas(),
            post_process_for_seg(
                keys=keys,
                preds=preds,
                score_th=m1,
                distance=distance,
            ).to_pandas(),
        )
        score2 = event_detection_ap(
            val_event_df.to_pandas(),
            post_process_for_seg(
                keys=keys,
                preds=preds,
                score_th=m2,
                distance=distance,
            ).to_pandas(),
        )
        if score1 >= score2:
            r = m2
            best_score = score1
            best_th = m1
        else:
            l = m1
            best_score = score2
            best_th = m2

        tqdm.write(f"score1(m1): {score1:.5f}({m1:.5f}), score2(m2): {score2:.5f}({m2:.5f}), l: {l:.5f}, r: {r:.5f}")

    return best_score, best_th
