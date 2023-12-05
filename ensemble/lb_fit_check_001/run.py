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
import optuna
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

from src.utils.metrics import event_detection_ap
from src.utils.detect_peak import post_process_from_2nd

import random

def random_sample_extract(series_ids, seed=0, debug=False):
    random.seed(seed)
    all_series_ids = random.sample(series_ids, 200)
    private_series_ids = all_series_ids[:150]
    public_series_ids = all_series_ids[150:] 
    if debug:
        private_series_ids = private_series_ids[:10]
        public_series_ids = public_series_ids[:10]
    return private_series_ids, public_series_ids


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")
LOGGER = logging.getLogger(Path(__file__).name)


def load_sakami_pred(cfg, name):
    LOGGER.info(f"load {name} pred")
    pred_df = pl.concat(
            [ 
                pl.read_parquet(Path(cfg.dir.pred_path) / "009.parquet",columns=['series_id', 'step', 'timestamp']),
                pl.read_parquet(Path(cfg.dir.pred_path) / f'sakami/{name}/valid_preds.parquet')
            ], how="horizontal"
        ).with_columns(
            pl.col("step").cast(pl.UInt32)
        ).rename({
            "stacking_prediction_onset": f"{name}_stacking_prediction_onset",
            "stacking_prediction_wakeup": f"{name}_stacking_prediction_wakeup",
            })
    return pred_df.select(["series_id", "step", f"{name}_stacking_prediction_onset", f"{name}_stacking_prediction_wakeup"])

def load_shimacos_nn_pred(cfg, name: str):
    LOGGER.info(f"load {name} pred")
    pred_df = (
        pl.concat(
            [
                pl.read_parquet(Path(cfg.dir.pred_path) / f"shimacos/{name}/fold{i}/result/valid.parquet") for i in range(5)
            ],
            how="vertical"
        )
    ).with_columns(
        pl.col("step").cast(pl.UInt32)
    ).with_columns(
            pl.col("step").cast(pl.UInt32)
    ).rename({
        "label_onset_pred": f"{name}_stacking_prediction_onset",
        "label_wakeup_pred": f"{name}_stacking_prediction_wakeup",}
    )
    return pred_df.select(["series_id", "step", f"{name}_stacking_prediction_onset", f"{name}_stacking_prediction_wakeup"])

def load_shimacos_pred(cfg, name: str):
    LOGGER.info(f"load {name} pred")
    pred_df = (
        pl.read_parquet(Path(cfg.dir.pred_path) / f"shimacos/{name}/result/pred_onset.parquet")
        .rename({"label_pred": f"{name}_stacking_prediction_onset"})
        .drop("label")
        .join(
            pl.read_parquet(Path(cfg.dir.pred_path) / f"shimacos/{name}/result/pred_wakeup.parquet").rename(
                {"label_pred": f"{name}_stacking_prediction_wakeup"}).drop("label"),
            on=["series_id", "step"],
            how="left",
        ).with_columns(pl.col("step").cast(pl.UInt32))
    )
    return pred_df.select(["series_id", "step", f"{name}_stacking_prediction_onset", f"{name}_stacking_prediction_wakeup"])

def load_and_concat_shimacos_preds(cfg, train_df):
    # train_df に予測を結合する
    train_df = train_df.with_columns(pl.lit(0).alias("count"))

    for name in cfg.shimacos_nn_models:
        pred_df = load_shimacos_nn_pred(cfg, name)
        train_df = train_df.join(pred_df, on=['series_id', 'step'], how='outer')
        train_df = train_df.with_columns(pl.col("count") + pl.col(f"{name}_stacking_prediction_onset").is_not_null().cast(int)) # null でないものの数をカウント

    for name in cfg.shimacos_models:
        pred_df = load_shimacos_pred(cfg, name)
        train_df = train_df.join(pred_df, on=['series_id', 'step'], how='outer')
        train_df = train_df.with_columns(pl.col("count") + pl.col(f"{name}_stacking_prediction_onset").is_not_null().cast(int)) # null でないものの数をカウント

    for name in cfg.sakami_models:
        pred_df = load_sakami_pred(cfg, name)
        train_df = train_df.join(pred_df, on=['series_id', 'step'], how='outer')
        train_df = train_df.with_columns(pl.col("count") + pl.col(f"{name}_stacking_prediction_onset").is_not_null().cast(int)) # null でないものの数をカウント


    # countが0のものは除く
    train_df = train_df.filter(pl.col("count") > 0)

    # null を 0 にする
    train_df = train_df.fill_null(0.0)

    # 12stepごとにchunk_idを振る
    train_df = train_df.with_columns(
        ((pl.col("step") - pl.col("step").shift(1)) != 12)
        .cast(int)
        .cumsum()
        .over("series_id")
        .fill_null(0)
        .alias("chunk_id")
    ).with_columns(pl.col("step").cast(pl.UInt32))

    return train_df



def cal_score(name2weight, params, pred_df, event_df):
    pred_df = pred_df.with_columns(
        pl.sum_horizontal(
            [pl.col(f"{name}_stacking_prediction_onset") * weight for name, weight in name2weight.items()]).alias("stacking_prediction_onset"),
        pl.sum_horizontal(
            [pl.col(f"{name}_stacking_prediction_wakeup") * weight for name, weight in name2weight.items()]).alias("stacking_prediction_wakeup"),
    )
    sub_df = post_process_from_2nd(
        pred_df,
        later_date_max_sub_rate=None,
        daily_score_offset=params["daily_score_offset"],
        tqdm_disable=True,
    )
    score = event_detection_ap(
        event_df.to_pandas(),
        sub_df.to_pandas(),
        tqdm_disable=True,
    )
    return score, sub_df


def objective(trial, name2weight, train_df, event_df):
    params = {
        "daily_score_offset": trial.suggest_float("daily_score_offset", 0, 20, step=0.5),
    }
    score, _ = cal_score(name2weight, params, train_df, event_df)
    return score


def run_process(cfg,  name2weight, train_df, event_df, study_name):
    study = optuna.load_study(study_name=study_name, storage=cfg.sql_storage)
    study.optimize(lambda trial: objective(trial, name2weight, train_df, event_df), 
                   n_trials=cfg.optuna.n_trials // cfg.optuna.n_jobs,
                   )

def plot_histogram(data, filename="score.png"):
    # 統計値の計算
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    plt.hist(data, bins=10, color='blue', alpha=0.7)
    plt.title(f"mean:{mean:.4}, median:{median:.4}, std:{std_dev:.4}")
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(filename)
    plt.close()

@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg: DictConfig):  # type: ignore
    LOGGER.info('-'*20 + ' START ' + '-'*20)
    LOGGER.info({k: v for k, v in cfg.items() if "fold_" not in k})

    LOGGER.info("load data")
    event_df = pl.read_csv(Path(cfg.dir.data_dir) / "train_events.csv").drop_nulls()
    event_df = event_df.with_columns(pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"))

    train_df = pl.read_parquet(Path(cfg.dir.data_dir) / "train_series.parquet", columns=["series_id", "step", "timestamp"])
    train_df = train_df.with_columns(pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z")).filter(
        pl.col("step") % 12 == 0
    )
    pred_df = load_and_concat_shimacos_preds(cfg, train_df)


    model_names = [name for name in cfg.shimacos_models] + [name for name in cfg.shimacos_nn_models] + [name for name in cfg.sakami_models]

    LOGGER.info("start optuna")


    series_ids = event_df.get_column("series_id").unique(maintain_order=True).to_list()
    name2weight = OmegaConf.to_container(cfg.name2weight, resolve=True)

    optimum_scores = []
    public_offset_scores = []
    score_diffs = []
    offset_diffs = []

    for seed in range(cfg.seed, cfg.seed+cfg.n_seed):
        LOGGER.info("-" * 10 + f"Seed: {seed}" + "-" * 10)

        private_series_ids, public_series_ids = random_sample_extract(series_ids, seed=seed, debug=cfg.debug)

        private_df = pred_df.filter(pl.col("series_id").is_in(private_series_ids))
        private_event_df = event_df.filter(pl.col("series_id").is_in(private_series_ids))
        public_df = pred_df.filter(pl.col("series_id").is_in(public_series_ids))
        public_event_df = event_df.filter(pl.col("series_id").is_in(public_series_ids))

        # public で optuna
        study_name = f"{cfg.exp_name}_public_{seed}"
        if cfg.debug:
            study_name += "_debug"
        study = optuna.create_study(
            study_name=study_name, 
            storage=cfg.sql_storage,
            load_if_exists=cfg.optuna.load_if_exists,
            direction="maximize",
        )        
        # public_df optuna
        _ = Parallel(n_jobs=cfg.optuna.n_jobs)(
            delayed(run_process)(cfg, name2weight, public_df, public_event_df, study_name) for _ in range(cfg.optuna.n_jobs)
        )
        study = optuna.load_study(study_name=study_name, storage=cfg.sql_storage)
        public_daily_score_offset = study.best_trial.params["daily_score_offset"]

        # private で optuna
        study_name = f"{cfg.exp_name}_private_{seed}"
        if cfg.debug:
            study_name += "_debug"
        study = optuna.create_study(
            study_name=study_name, 
            storage=cfg.sql_storage,
            load_if_exists=cfg.optuna.load_if_exists,
            direction="maximize",
        )        
        # private_df optuna
        _ = Parallel(n_jobs=cfg.optuna.n_jobs)(
            delayed(run_process)(cfg, name2weight, private_df, private_event_df, study_name) for _ in range(cfg.optuna.n_jobs)
        )
        study = optuna.load_study(study_name=study_name, storage=cfg.sql_storage)
        private_daily_score_offset = study.best_trial.params["daily_score_offset"]

        optimum_score, _ = cal_score(name2weight, {"daily_score_offset": private_daily_score_offset}, private_df, private_event_df)
        public_offset_score, _ = cal_score(name2weight, {"daily_score_offset": public_daily_score_offset}, private_df, private_event_df)

        LOGGER.info(f"Optimum score: {optimum_score}, private daily_score_offset: {private_daily_score_offset}")
        LOGGER.info(f"Public offset score: {public_offset_score}, public daily_score_offset: {public_daily_score_offset}")
        
        optimum_scores.append(optimum_score)
        public_offset_scores.append(public_offset_score)
        score_diffs.append(optimum_score-public_offset_score)
        offset_diffs.append(private_daily_score_offset-public_daily_score_offset)

        plot_histogram(score_diffs, filename=f"score_diffs_{seed}.png")
        plot_histogram(offset_diffs, filename=f"offset_diffs_{seed}.png")

        if cfg.debug:
            break
    
    LOGGER.info("-" * 10 + "Result" + "-" * 10)
    # seed_scores_by_mean_params
    mean = np.mean(score_diffs)
    median = np.median(score_diffs)
    min_val = np.min(score_diffs)
    max_val = np.max(score_diffs)
    std = np.std(score_diffs)
    LOGGER.info(f"score_diffs statistics:  mean: {mean}, median: {median}, min: {min_val}, max: {max_val}, std: {std}")

    mean = np.mean(offset_diffs)
    median = np.median(offset_diffs)
    min_val = np.min(offset_diffs)
    max_val = np.max(offset_diffs)
    std = np.std(offset_diffs)
    LOGGER.info(f"offset_diffs statistics:  mean: {mean}, median: {median}, min: {min_val}, max: {max_val}, std: {std}")

    plot_histogram(score_diffs, filename="score_diffs.png")
    plot_histogram(offset_diffs, filename="offset_diffs.png")

    LOGGER.info('-'*20 + ' END ' + '-'*20)

if __name__ == "__main__":
    main()
