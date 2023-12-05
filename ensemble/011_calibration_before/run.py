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
from pytorch_lightning import Trainer, seed_everything
import pickle
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold

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


def daily_normalize_df(train_df, name, daily_score_offset, event2offset: dict[str, str] = {"onset": "5h", "wakeup": "0h"}):
    for event, offset in event2offset.items():
        event_pred_col = f"{name}_stacking_prediction_{event}" 
        train_df = (
            train_df.with_columns(
                pl.col("timestamp").dt.offset_by(offset).dt.date().alias("date")
            )
            .with_columns(pl.col(event_pred_col).sum().over(["series_id", "date"]).alias("date_sum"))
            .with_columns(
                pl.col(event_pred_col) / (pl.col("date_sum") + (1 / (daily_score_offset + pl.col("date_sum"))))
            ).drop(["date_sum", "date"])
        )
    return train_df

def calibrate_df(cfg, pred_df, name):
    LOGGER.info(f"calibrate {name} pred")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    for event in ["onset", "wakeup"]:
        event_pred_col = f"{name}_stacking_prediction_{event}" 
        oof = np.zeros(len(pred_df))
        # row_id を付与
        pred_df = pred_df.with_columns(pl.arange(0, len(pred_df)).alias("row_id"))
        # クロスバリデーションでキャリブレーションを行う。作成したモデルは保存する        
        for fold in range(cfg.n_fold):
            train_df = pred_df.filter(pl.col("series_id").is_in(cfg[f"fold_{fold}"].train_series_ids))
            valid_df = pred_df.filter(pl.col("series_id").is_in(cfg[f"fold_{fold}"].valid_series_ids))

            model_path = f"{model_dir}/{name}_{event}_calibrator_fold{fold}.pkl"

            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    calibrator = pickle.load(f)
            else:            
                prob = train_df.get_column(event_pred_col).to_numpy()
                X = np.zeros((len(prob), 2))
                X[:, 0] = 1-prob
                X[:, 1] = prob
                y = train_df.get_column(f"label_{event}").to_numpy()
                
                calibrator = CalibratedClassifierCV(
                    cv=cfg.calibration.cv,
                    method=cfg.calibration.method,
                )
                calibrator.fit(
                    X,
                    y,
                )
                with open(path, "wb") as f:
                    pickle.dump(calibrator, f)

            X_val = np.zeros((len(valid_df), 2))
            X_val[:, 0] = 1-valid_df.get_column(event_pred_col).to_numpy()
            X_val[:, 1] = valid_df.get_column(event_pred_col).to_numpy()
            oof[valid_df.get_column("row_id").to_numpy()] = calibrator.predict_proba(
                X_val
            )[:, 1]
        
        pred_df = pred_df.with_columns(
            pl.Series(oof).alias(event_pred_col)
        ) 
    return pred_df

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



def cal_score(cfg, name2weight, params, df, event_df):

    # normalize
    pred_df = df.clone()
    for name in cfg.shimacos_nn_models + cfg.shimacos_models + cfg.sakami_models:
        pred_df = daily_normalize_df(pred_df, name, params['daily_score_offset'])

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


def objective(trial, cfg, names, train_df, event_df):
    weights = [trial.suggest_float(name, 0, 1) for name in names]
    params = {
        "daily_score_offset": trial.suggest_float("daily_score_offset", 0, 20),
    }
    weights = np.array(weights) / np.sum(weights)
    score = cal_score(cfg, dict(zip(names, weights)), params, train_df, event_df)
    return score


def run_process(cfg,  names, train_df, event_df, study_name):
    study = optuna.load_study(study_name=study_name, storage=cfg.sql_storage)
    study.optimize(lambda trial: objective(trial, cfg, names, train_df, event_df), 
                   n_trials=cfg.optuna.n_trials // cfg.optuna.n_jobs,
                   )

@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg: DictConfig):  # type: ignore
    LOGGER.info('-'*10 + ' START ' + '-'*10)
    LOGGER.info({k: v for k, v in cfg.items() if "fold_" not in k})
    seed_everything(cfg.seed)

    LOGGER.info("load data")
    event_df = pl.read_csv(Path(cfg.dir.data_dir) / "train_events.csv").drop_nulls()
    event_df = event_df.with_columns([pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"), pl.col("step").cast(pl.UInt32)])

    train_df = pl.read_parquet(Path(cfg.dir.data_dir) / "train_series.parquet", columns=["series_id", "step", "timestamp"])
    train_df = train_df.with_columns(pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z")).filter(
        pl.col("step") % 12 == 0
    )
    # event をラベルとして結合
    train_df = train_df.join(event_df.select(["series_id", "step", "event"]), on=["series_id", "step"], how="left")
    # onset, wakeup をラベルとする
    train_df = train_df.with_columns(
        pl.col("event").str.contains("onset").cast(pl.Int8).alias("label_onset").fill_null(0),
        pl.col("event").str.contains("wakeup").cast(pl.Int8).alias("label_wakeup").fill_null(0),
    )

    pred_df = load_and_concat_shimacos_preds(cfg, train_df)

    model_names = cfg.shimacos_models + cfg.shimacos_nn_models + cfg.sakami_models

    for name in model_names:
        pred_df = calibrate_df(cfg, pred_df, name)


    LOGGER.info("start optuna")

    model_weights_list = []
    params_list = []
    scores = []
    for fold in range(cfg.n_fold):
        LOGGER.info("-" * 10 + f"Fold: {fold}" + "-" * 10)

        # データ分割
        train_df = pred_df.filter(pl.col("series_id").is_in(cfg[f"fold_{fold}"].train_series_ids))
        train_event_df = event_df.filter(pl.col("series_id").is_in(cfg[f"fold_{fold}"].train_series_ids))
        valid_df = pred_df.filter(pl.col("series_id").is_in(cfg[f"fold_{fold}"].valid_series_ids))
        valid_event_df = event_df.filter(pl.col("series_id").is_in(cfg[f"fold_{fold}"].valid_series_ids))

        # debug
        if cfg.debug:
            num = 10
            series_ids = train_df.get_column("series_id").unique().to_list()[:num]
            train_df = train_df.filter(pl.col("series_id").is_in(series_ids))            
            train_event_df = train_event_df.filter(pl.col("series_id").is_in(series_ids))

        study_name = f"{cfg.exp_name}_{fold}_debug" if cfg.debug else f"{cfg.exp_name}_{fold}"
        study = optuna.create_study(
            study_name=study_name, 
            storage=cfg.sql_storage,
            load_if_exists=cfg.optuna.load_if_exists,
            direction="maximize",
        )
        
        # train optuna
        _ = Parallel(n_jobs=cfg.optuna.n_jobs)(
            delayed(run_process)(cfg, model_names, train_df, train_event_df, study_name) for _ in range(cfg.optuna.n_jobs)
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

        # valid        
        score = cal_score(cfg, model_weights, params, valid_df, valid_event_df)

        model_weights_list.append(model_weights)
        params_list.append(params)
        scores.append(score)

        LOGGER.info(f"Fold: {fold},  Best params: {params}, Valid score: {score}")
        LOGGER.info(f"Fold: {fold},  Best model_weights: {model_weights}")

    mean_best_model_weights = {}
    for name in model_names:
        mean_best_model_weights[name] = np.mean([model_weights[name] for model_weights in model_weights_list])
    mean_best_params = {}
    for k in params_list[0].keys():
        mean_best_params[k] = np.mean([params[k] for params in params_list])

    LOGGER.info(f"Mean best model_weights: {mean_best_model_weights}")
    LOGGER.info(f"Mean best params: {mean_best_params}")
    LOGGER.info(f"Mean score: {np.mean(scores)}")
    score = cal_score(cfg, mean_best_model_weights, mean_best_params, pred_df, event_df)
    LOGGER.info(f"OOF score: {score}")



    LOGGER.info('-'*10 + ' END ' + '-'*10)

if __name__ == "__main__":
    main()
