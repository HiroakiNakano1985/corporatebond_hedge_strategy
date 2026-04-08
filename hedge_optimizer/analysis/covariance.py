"""共分散行列推定モジュール

3ファクター [ΔR_USD, Δ(USDJPY), ΔI_spread] の共分散行列を推定する。
- ローリングウィンドウ（デフォルト52週）
- EWMA（λ=0.94）
"""

import numpy as np
import pandas as pd

from hedge_optimizer.config import EWMA_LAMBDA, WINDOW_WEEKS

FACTOR_COLS = ["d_rate_usd", "d_usdjpy", "d_i_spread"]


def rolling_covariance(
    changes: pd.DataFrame,
    window: int = WINDOW_WEEKS,
) -> pd.DataFrame:
    """ローリングウィンドウで共分散行列を推定する。

    Returns:
        各時点での3×3共分散行列を格納したDataFrame。
        列は 'cov_ij' 形式（例: cov_00, cov_01, ...）
    """
    factors = changes[FACTOR_COLS].values
    n = len(factors)
    dates = changes.index

    results = []
    for t in range(window, n + 1):
        sub = factors[t - window : t]
        cov = np.cov(sub, rowvar=False, ddof=1)
        row = {"date": dates[t - 1]}
        for i in range(3):
            for j in range(3):
                row[f"cov_{i}{j}"] = cov[i, j]
        results.append(row)

    df = pd.DataFrame(results).set_index("date")
    return df


def ewma_covariance(
    changes: pd.DataFrame,
    lam: float = EWMA_LAMBDA,
    min_periods: int = WINDOW_WEEKS,
) -> pd.DataFrame:
    """EWMAで共分散行列を推定する。

    S_t = λ * S_{t-1} + (1-λ) * r_t * r_t'
    """
    factors = changes[FACTOR_COLS].values
    n = len(factors)
    dates = changes.index

    # 初期化: 最初のmin_periods分のサンプル共分散
    init_data = factors[:min_periods]
    S = np.cov(init_data, rowvar=False, ddof=1)

    results = []

    # min_periods以前はスキップ
    for t in range(min_periods, n):
        r = factors[t].reshape(-1, 1)
        # EWMA更新
        S = lam * S + (1 - lam) * (r @ r.T)

        row = {"date": dates[t]}
        for i in range(3):
            for j in range(3):
                row[f"cov_{i}{j}"] = S[i, j]
        results.append(row)

    df = pd.DataFrame(results).set_index("date")
    return df


def get_latest_covariance(
    changes: pd.DataFrame,
    method: str = "rolling",
) -> np.ndarray:
    """最新時点の3×3共分散行列を返す。

    Args:
        method: "rolling" or "ewma"

    Returns:
        3×3 numpy array
    """
    if method == "rolling":
        cov_df = rolling_covariance(changes)
    elif method == "ewma":
        cov_df = ewma_covariance(changes)
    else:
        raise ValueError(f"Unknown method: {method}")

    last_row = cov_df.iloc[-1]
    cov_matrix = np.array([
        [last_row["cov_00"], last_row["cov_01"], last_row["cov_02"]],
        [last_row["cov_10"], last_row["cov_11"], last_row["cov_12"]],
        [last_row["cov_20"], last_row["cov_21"], last_row["cov_22"]],
    ])
    return cov_matrix


def print_covariance_summary(cov_matrix: np.ndarray) -> None:
    """共分散行列のサマリーを表示する。"""
    labels = ["ΔR_USD", "Δ(USDJPY)", "ΔI_spread"]

    print("\n共分散行列:")
    header = "           " + "  ".join(f"{l:>12s}" for l in labels)
    print(header)
    for i, label in enumerate(labels):
        vals = "  ".join(f"{cov_matrix[i, j]:12.6f}" for j in range(3))
        print(f"{label:>12s}  {vals}")

    # 相関行列
    std = np.sqrt(np.diag(cov_matrix))
    corr = cov_matrix / np.outer(std, std)

    print("\n相関行列:")
    print(header)
    for i, label in enumerate(labels):
        vals = "  ".join(f"{corr[i, j]:12.4f}" for j in range(3))
        print(f"{label:>12s}  {vals}")

    print(f"\n標準偏差（週次）:")
    for label, s in zip(labels, std):
        print(f"  {label}: {s:.4f}")

    # 年率換算（√52）
    annual = std * np.sqrt(52)
    print(f"\n標準偏差（年率換算）:")
    for label, s in zip(labels, annual):
        print(f"  {label}: {s:.4f}")
