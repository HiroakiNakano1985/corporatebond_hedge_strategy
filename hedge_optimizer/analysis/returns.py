"""リターンモデルモジュール

ポートフォリオの円建てリターンを計算する。

■ 円建てリターン（実現値）:

  r_p = I_spread/100
      + (1 - h_ir) * Swap_rate       ... 未ヘッジ: 固定金利キャリー
      + h_ir * SOFR                   ... ヘッジ済: スワップで変動受取
      + (1 - h_fx) * Δ(USDJPY)       ... 未ヘッジ為替リスク
      - h_fx * FX_hedge_cost          ... 為替ヘッジコスト
      - (1 - h_ir) * D * ΔR_USD      ... 金利変動による価格変化
      - D_spread * ΔI_spread / 100    ... スプレッド変動による価格変化

■ 期待リターン（年率）:

  E[r_p] = I_spread/100
         + (1 - h_ir) * Swap_rate
         + h_ir * SOFR
         + (1 - h_fx) * E[Δ(USDJPY)]   ... ドル円期待変動
         - h_fx * FX_hedge_cost

  ※ E[ΔR_USD] = 0, E[ΔI_spread] = 0 と仮定
  ※ E[Δ(USDJPY)] はパラメータ（デフォルト=0: ランダムウォーク仮定）

■ リスクファクター（週次変動）:

  f1 = ΔR_USD      （米金利変化、%ポイント）
  f2 = Δ(USDJPY)   （為替変化、単純リターン）
  f3 = ΔI_spread   （スプレッド変化、bp）

■ ポートフォリオ分散:

  σ²_p = w' * Σ * w
  w = [-(1-h_ir)*D, (1-h_fx), -D_spread/100]
"""

import numpy as np

from hedge_optimizer.config import DURATION, DURATION_SPREAD


def expected_return(
    e_i_spread: float,
    swap_rate: float,
    sofr: float,
    fx_hedge_cost: float,
    h_fx: float,
    h_ir: float,
    e_fx_return: float = 0.0,
) -> float:
    """期待リターン（年率%）を計算する。

    E[r_p] = I_spread/100
           + (1-h_ir)*Swap_rate + h_ir*SOFR
           + (1-h_fx)*E[Δ(USDJPY)]
           - h_fx*FX_hedge_cost

    Args:
        e_i_spread: 期待I_spread（bp）
        swap_rate: スワップレート（%）
        sofr: SOFR短期金利（%）
        fx_hedge_cost: 為替ヘッジコスト（%） = SOFR_3M - TONA + CIP_basis
        h_fx: 為替ヘッジ比率 [0, 1]
        h_ir: 金利ヘッジ比率 [0, 1]
        e_fx_return: ドル円の期待変動（年率%）。0=ランダムウォーク

    Returns:
        年率リターン（%）
    """
    return (
        e_i_spread / 100.0                  # I_spread: bp → %
        + (1 - h_ir) * swap_rate            # 未ヘッジ: 固定金利キャリー
        + h_ir * sofr                       # ヘッジ済: SOFR受取
        + (1 - h_fx) * e_fx_return          # 未ヘッジ為替の期待変動
        - h_fx * fx_hedge_cost              # FXヘッジコスト
    )


def portfolio_variance(
    h_fx: float,
    h_ir: float,
    cov_matrix: np.ndarray,
    D: float = DURATION,
    D_spread: float = DURATION_SPREAD,
) -> float:
    """ポートフォリオ分散を計算する。

    w = [-(1-h_ir)*D, (1-h_fx), -D_spread/100]
    σ²_p = w' * Σ * w

    Args:
        cov_matrix: 3×3共分散行列 [ΔR_USD, Δ(USDJPY), ΔI_spread]
            - ΔR_USD: %ポイント単位
            - Δ(USDJPY): 単純リターン
            - ΔI_spread: bp単位

    Returns:
        ポートフォリオ分散（週次）
    """
    w = np.array([
        -(1 - h_ir) * D,           # 金利: 利回り上昇→価格下落
        (1 - h_fx),                 # 為替: 円安→プラス
        -D_spread / 100.0,          # スプレッド: 拡大→価格下落（bp→%変換）
    ])
    return w @ cov_matrix @ w


def portfolio_std(
    h_fx: float,
    h_ir: float,
    cov_matrix: np.ndarray,
    D: float = DURATION,
    D_spread: float = DURATION_SPREAD,
    annualize: bool = True,
) -> float:
    """ポートフォリオ標準偏差を計算する。

    Args:
        annualize: Trueの場合、年率換算（×√52）

    Returns:
        ポートフォリオ標準偏差（%）
    """
    var = portfolio_variance(h_fx, h_ir, cov_matrix, D, D_spread)
    std = np.sqrt(max(var, 0.0))
    if annualize:
        std *= np.sqrt(52)
    return std * 100  # %表示


def sharpe_ratio(
    h_fx: float,
    h_ir: float,
    cov_matrix: np.ndarray,
    e_i_spread: float,
    swap_rate: float,
    sofr: float,
    fx_hedge_cost: float,
    D: float = DURATION,
    D_spread: float = DURATION_SPREAD,
    risk_free: float = 0.0,
    e_fx_return: float = 0.0,
) -> float:
    """シャープレシオを計算する。

    Sharpe = (E[r_p] - risk_free) / σ_p

    Returns:
        シャープレシオ（年率ベース）
    """
    er = expected_return(
        e_i_spread, swap_rate, sofr, fx_hedge_cost, h_fx, h_ir, e_fx_return,
    )
    std = portfolio_std(h_fx, h_ir, cov_matrix, D, D_spread, annualize=True)

    if std < 1e-10:
        return 0.0

    return (er - risk_free) / std
