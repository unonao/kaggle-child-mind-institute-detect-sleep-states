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

from src.utils.metrics import event_detection_ap
from src.utils.detect_peak import post_process_from_2nd

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
        train_df = train_df.with_columns(pl.col("count") + pl.col(f"{name}_stacking_prediction_onset").is_not_null().cast(int))

    for name in cfg.shimacos_models:
        pred_df = load_shimacos_pred(cfg, name)
        train_df = train_df.join(pred_df, on=['series_id', 'step'], how='outer')
        train_df = train_df.with_columns(pl.col("count") + pl.col(f"{name}_stacking_prediction_onset").is_not_null().cast(int))

    for name in cfg.sakami_models:
        pred_df = load_sakami_pred(cfg, name)
        train_df = train_df.join(pred_df, on=['series_id', 'step'], how='outer')
        train_df = train_df.with_columns(pl.col("count") + pl.col(f"{name}_stacking_prediction_onset").is_not_null().cast(int))


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
    return score


def objective(trial, names, train_df, event_df):
    weights = [trial.suggest_float(name, 0, 1) for name in names]
    params = {
        "daily_score_offset": trial.suggest_float("daily_score_offset", 0.0, 20.0),
    }
    weights = np.array(weights) / np.sum(weights)
    score = cal_score(dict(zip(names, weights)), params, train_df, event_df)
    return score


def run_process(cfg,  names, train_df, event_df, study_name):
    study = optuna.load_study(study_name=study_name, storage=cfg.sql_storage)
    study.optimize(lambda trial: objective(trial, names, train_df, event_df), 
                   n_trials=cfg.optuna.n_trials // cfg.optuna.n_jobs,
                   )

@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg: DictConfig):  # type: ignore
    LOGGER.info('-'*10 + ' START ' + '-'*10)
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

    # debug
    if cfg.debug:
        num = 10
        series_ids = pred_df.get_column("series_id").unique().to_list()[:num]
        pred_df = pred_df.filter(pl.col("series_id").is_in(series_ids))            
        event_df = event_df.filter(pl.col("series_id").is_in(series_ids))


    study_name = f"{cfg.exp_name}_debug" if cfg.debug else f"{cfg.exp_name}"
    study = optuna.create_study(
        study_name=study_name, 
        storage=cfg.sql_storage,
        load_if_exists=cfg.optuna.load_if_exists,
        direction="maximize",
    )
    
    # train optuna
    _ = Parallel(n_jobs=cfg.optuna.n_jobs)(
        delayed(run_process)(cfg, model_names, pred_df, event_df, study_name) for _ in range(cfg.optuna.n_jobs)
    )
    study = optuna.load_study(study_name=study_name, storage=cfg.sql_storage)

    model_weights = {}
    params = {}
    for k,v in study.best_trial.params.items():
        if k in model_names:
            model_weights[k] = v
        else:
            params[k] = v
    model_weights = {name: weight / np.sum(list(model_weights.values())) for name, weight in model_weights.items()}

    LOGGER.info(f"Best train value: {study.best_value}, Best model_weights: {model_weights}, Best params: {params}")

    LOGGER.info('-'*10 + ' END ' + '-'*10)

if __name__ == "__main__":
    main()
