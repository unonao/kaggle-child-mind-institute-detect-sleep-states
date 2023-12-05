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
    return score, sub_df


def objective(cfg,trial, names, train_df, event_df):
    weights = [trial.suggest_float(name, 0, 1, step=0.1) for name in names]
    params = {
        "daily_score_offset": trial.suggest_float("daily_score_offset", 0, 20, step=0.5),
    }
    weights = np.array(weights) / np.sum(weights)
    score, _ = cal_score(cfg,dict(zip(names, weights)), params, train_df, event_df)
    return score


def run_process(cfg,  names, train_df, event_df, study_name):
    study = optuna.load_study(study_name=study_name, storage=cfg.sql_storage)
    study.optimize(lambda trial: objective(cfg,trial, names, train_df, event_df), 
                   n_trials=cfg.optuna.n_trials // cfg.optuna.n_jobs,
                   )

def plot_histogram(data, filename="score.png"):
    # 統計値の計算
    mean = np.mean(data)
    median = np.median(data)
    min_val = np.min(data)
    max_val = np.max(data)
    std_dev = np.std(data)

    plt.hist(data, bins=10, color='blue', alpha=0.7)
    plt.title(f"mean:{mean:.4}, median:{median:.4}, min:{min_val:.4}, max:{max_val:.4}, std:{std_dev:.4}")
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


    seed_model_weights_list = []
    seed_params_list = []
    seed_scores_by_mean_params = []
    seed_all_scores = []

    series_ids = event_df.get_column("series_id").unique(maintain_order=True).to_list()

    for seed in range(cfg.seed, cfg.seed+cfg.n_seed):
        LOGGER.info("-" * 10 + f"Seed: {seed}" + "-" * 10)
        model_weights_list = []
        params_list = []
        scores = []
        kf = KFold(n_splits=cfg.n_fold, shuffle=True, random_state=seed)
        for fold, (train_idx, valid_idx) in enumerate(kf.split(series_ids)):
            LOGGER.info("-" * 5 + f"Fold: {fold}" + "-" * 5)

            train_series_ids = [series_ids[i] for i in train_idx]
            valid_series_ids = [series_ids[i] for i in valid_idx]
            train_df = pred_df.filter(pl.col("series_id").is_in(train_series_ids))
            train_event_df = event_df.filter(pl.col("series_id").is_in(train_series_ids))
            valid_df = pred_df.filter(pl.col("series_id").is_in(valid_series_ids))
            valid_event_df = event_df.filter(pl.col("series_id").is_in(valid_series_ids))

            # debug
            if cfg.debug:
                num = 10
                few_series_ids = train_df.get_column("series_id").unique().to_list()[:num]
                train_df = train_df.filter(pl.col("series_id").is_in(few_series_ids))            
                train_event_df = train_event_df.filter(pl.col("series_id").is_in(few_series_ids))

            study_name = f"{cfg.exp_name}_{seed}_{fold}"
            if cfg.debug:
                study_name += "_debug"

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
            score, _ = cal_score(cfg,model_weights, params, valid_df, valid_event_df)

            model_weights_list.append(model_weights)
            seed_model_weights_list.append(model_weights)
            params_list.append(params)
            seed_params_list.append(params)
            scores.append(score)
            seed_all_scores.append(score)
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
        score, _ = cal_score(cfg,mean_best_model_weights, mean_best_params, pred_df, event_df)
        LOGGER.info(f"Score by mean params: {score}")
        seed_scores_by_mean_params.append(score)

        plot_histogram(seed_all_scores, filename=f"score_{seed}.png")
        for name in model_names:
            plot_histogram([model_weights[name] for model_weights in seed_model_weights_list], filename=f"weight_{name}_{seed}.png")
        for k in params_list[0].keys():
            plot_histogram([params[k] for params in seed_params_list], filename=f"param_{k}_{seed}.png")
        if cfg.debug:
            break
    
    LOGGER.info("-" * 10 + "Result" + "-" * 10)
    mean_best_model_weights = {}
    for name in model_names:
        mean_best_model_weights[name] = np.mean([model_weights[name] for model_weights in seed_model_weights_list])
    mean_best_params = {}
    for k in params_list[0].keys():
        mean_best_params[k] = np.mean([params[k] for params in seed_params_list])
    

    # seed_scores_by_mean_params
    mean = np.mean(seed_scores_by_mean_params)
    median = np.median(seed_scores_by_mean_params)
    min_val = np.min(seed_scores_by_mean_params)
    max_val = np.max(seed_scores_by_mean_params)
    std = np.std(seed_scores_by_mean_params)
    LOGGER.info(f"Score by mean params statistics:  mean: {mean}, median: {median}, min: {min_val}, max: {max_val}, std: {std}")

    # seed_all_scores
    mean = np.mean(seed_all_scores)
    median = np.median(seed_all_scores)
    min_val = np.min(seed_all_scores)
    max_val = np.max(seed_all_scores)
    std = np.std(seed_all_scores)

    LOGGER.info(f"All score statistics:  mean: {mean}, median: {median}, min: {min_val}, max: {max_val}, std: {std}")
    LOGGER.info(f"Mean best model_weights: {mean_best_model_weights}")
    LOGGER.info(f"Mean best params: {mean_best_params}")
    
    plot_histogram(seed_all_scores)

    LOGGER.info('-'*20 + ' END ' + '-'*20)

if __name__ == "__main__":
    main()
