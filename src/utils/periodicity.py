from pathlib import Path

import numpy as np
from omegaconf import DictConfig


def downsample(sequence, factor=10):
    """
    Downsamples the sequence by the given factor.
    """
    return sequence[::factor]


def resize_1d_array(array, new_size):
    """
    Resizes a 1D numpy array to a new size using interpolation.
    """
    return np.interp(np.linspace(0, len(array) - 1, new_size), np.arange(len(array)), array)


def predict_periodicity(seq: np.ndarray, downsample_rate: int = 15, split_hour: int = 4) -> np.ndarray:
    """
    split_hourごとにフレームに分割して、同じ波形が現れたフレームは周期性ありとみなす

    Args:
        seq (np.ndarray): 1D array of shape (n,)
        downsample_rate (int, optional): 小さいほど間違ったものを検出しにくくなるが遅くなる
        split_hour (int, optional):  周期性の検出期間は split_hour ごとに行われる。24 の約数であること。大きいほど間違ったものを検出しにくくなる。
    Returns:
        pred (np.ndarray): 1D array of shape (n,)
    """

    # seq をダウンサンプリングして seq_downsampled に
    seq_downsampled = downsample(seq, downsample_rate)

    # seq_downsampled を split_hour ごとに分割した chunks (chunk_num, d) を作る（足りない部分は0埋め）
    split_step = split_hour * 3600 // 5 // downsample_rate
    valid_length = ((len(seq_downsampled) + (split_step - 1)) // split_step) * split_step  # split_step に合うように
    seq_downsampled_padded = np.zeros(valid_length)
    seq_downsampled_padded[: len(seq_downsampled)] = seq_downsampled
    chunks = seq_downsampled_padded.reshape(-1, split_step)

    # 各ベクトルを正規化し chunks・chunks.T で (chunk_num,chunk_num) のcos類似度を求め、対角線上を0にした後にmaxを取って chunk_num サイズの予測 pred_chunk を得る
    norm_vecs = chunks / np.linalg.norm(chunks, axis=1, keepdims=True)
    cosine_sim_matrix = np.dot(norm_vecs, norm_vecs.T)
    cosine_sim_matrix[range(len(cosine_sim_matrix)), range(len(cosine_sim_matrix))] = 0
    pred_chunk = cosine_sim_matrix.max(axis=0) > 0.99

    # 最後の一個前が true なら、最後もtrueにする（最後は0埋めしたのでうまくできていない）
    pred_chunk[-1] = pred_chunk[-2:-1].max()

    # pred_vecを元のsequenceのサイズに戻す
    pred = resize_1d_array(pred_chunk.repeat(split_step)[: len(seq_downsampled)], len(seq))
    return pred


def get_periodicity_dict(cfg: DictConfig) -> dict[np.ndarray]:
    phase = cfg.phase if "phase" in cfg else "train"
    feature_dir = Path(cfg.dir.processed_dir) / phase
    series_ids = [x.name for x in feature_dir.glob("*")]
    periodicity_dict = {}
    for series_id in series_ids:
        seq = np.load(feature_dir / series_id / "enmo.npy")
        periodicity_dict[series_id] = predict_periodicity(seq)
    return periodicity_dict
