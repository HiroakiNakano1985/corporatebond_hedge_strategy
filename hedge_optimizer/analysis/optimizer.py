"""最適化モジュール

Grid Search + SciPy minimize によるシャープレシオ最大化。
"""

import numpy as np
from scipy.optimize import minimize

from hedge_optimizer.analysis.returns import (
    expected_return,
    portfolio_std,
    sharpe_ratio,
)
from hedge_optimizer.config import (
    DURATION,
    DURATION_SPREAD,
    H_GRID_SIZE,
    RISK_FREE_RATE,
)


def grid_search(
    cov_matrix: np.ndarray,
    e_i_spread: float,
    swap_rate: float,
    sofr: float,
    fx_hedge_cost: float,
    e_fx_return: float = 0.0,
    D: float = DURATION,
    D_spread: float = DURATION_SPREAD,
    risk_free: float = RISK_FREE_RATE,
    grid_size: int = H_GRID_SIZE,
) -> dict:
    """Grid Searchによる最適ヘッジ比率探索。

    Returns:
        dict with keys:
        - h_fx_grid, h_ir_grid: meshgrid
        - sharpe_grid, er_grid, std_grid: 2D配列
        - opt_h_fx, opt_h_ir: 最適ヘッジ比率
        - opt_sharpe, opt_er, opt_std: 最適点の値
    """
    h_fx_vals = np.linspace(0, 1, grid_size)
    h_ir_vals = np.linspace(0, 1, grid_size)
    h_fx_grid, h_ir_grid = np.meshgrid(h_fx_vals, h_ir_vals)

    sharpe_grid = np.zeros_like(h_fx_grid)
    er_grid = np.zeros_like(h_fx_grid)
    std_grid = np.zeros_like(h_fx_grid)

    for i in range(grid_size):
        for j in range(grid_size):
            hfx = h_fx_grid[i, j]
            hir = h_ir_grid[i, j]

            er_grid[i, j] = expected_return(
                e_i_spread, swap_rate, sofr, fx_hedge_cost, hfx, hir,
                e_fx_return,
            )
            std_grid[i, j] = portfolio_std(
                hfx, hir, cov_matrix, D, D_spread, annualize=True,
            )
            sharpe_grid[i, j] = sharpe_ratio(
                hfx, hir, cov_matrix,
                e_i_spread, swap_rate, sofr, fx_hedge_cost,
                D, D_spread, risk_free, e_fx_return,
            )

    # 最適点
    idx = np.unravel_index(np.argmax(sharpe_grid), sharpe_grid.shape)
    opt_h_fx = h_fx_grid[idx]
    opt_h_ir = h_ir_grid[idx]

    return {
        "h_fx_grid": h_fx_grid,
        "h_ir_grid": h_ir_grid,
        "h_fx_vals": h_fx_vals,
        "h_ir_vals": h_ir_vals,
        "sharpe_grid": sharpe_grid,
        "er_grid": er_grid,
        "std_grid": std_grid,
        "opt_h_fx": opt_h_fx,
        "opt_h_ir": opt_h_ir,
        "opt_sharpe": sharpe_grid[idx],
        "opt_er": er_grid[idx],
        "opt_std": std_grid[idx],
    }


def scipy_optimize(
    cov_matrix: np.ndarray,
    e_i_spread: float,
    swap_rate: float,
    sofr: float,
    fx_hedge_cost: float,
    e_fx_return: float = 0.0,
    D: float = DURATION,
    D_spread: float = DURATION_SPREAD,
    risk_free: float = RISK_FREE_RATE,
    x0: tuple[float, float] | None = None,
) -> dict:
    """SciPy minimizeによる精緻化最適化。

    Args:
        x0: 初期値 (h_fx, h_ir)。Noneの場合は (0.5, 0.5)。

    Returns:
        dict with keys: opt_h_fx, opt_h_ir, opt_sharpe, opt_er, opt_std, result
    """
    if x0 is None:
        x0 = (0.5, 0.5)

    def neg_sharpe(x):
        return -sharpe_ratio(
            x[0], x[1], cov_matrix,
            e_i_spread, swap_rate, sofr, fx_hedge_cost,
            D, D_spread, risk_free, e_fx_return,
        )

    bounds = [(0, 1), (0, 1)]
    result = minimize(neg_sharpe, x0, method="L-BFGS-B", bounds=bounds)

    opt_h_fx, opt_h_ir = result.x
    opt_sr = -result.fun
    opt_er = expected_return(
        e_i_spread, swap_rate, sofr, fx_hedge_cost, opt_h_fx, opt_h_ir,
        e_fx_return,
    )
    opt_std = portfolio_std(opt_h_fx, opt_h_ir, cov_matrix, D, D_spread)

    return {
        "opt_h_fx": opt_h_fx,
        "opt_h_ir": opt_h_ir,
        "opt_sharpe": opt_sr,
        "opt_er": opt_er,
        "opt_std": opt_std,
        "result": result,
    }


def sensitivity_analysis(
    cov_matrix: np.ndarray,
    e_i_spread: float,
    swap_rate: float,
    sofr: float,
    sofr_90d: float,
    japan_call: float,
    fx_hedge_cost_base: float,
    e_fx_return: float = 0.0,
    D_base: float = DURATION,
    D_spread: float = DURATION_SPREAD,
    risk_free: float = RISK_FREE_RATE,
) -> dict:
    """感応度分析を実行する。

    - CIPベーシス: -0.1%〜-0.8%
    - デュレーション: 4年〜9年
    - ヘッジコスト水準の変化

    Returns:
        dict with analysis results
    """
    results = {}

    # 1. CIPベーシス感応度
    cip_basis_range = np.arange(-0.001, -0.009, -0.001)  # -0.1%〜-0.8%
    cip_results = []
    for cip in cip_basis_range:
        fx_hc = sofr_90d - japan_call + cip * 100
        opt = scipy_optimize(
            cov_matrix, e_i_spread, swap_rate, sofr, fx_hc, e_fx_return,
            D_base, D_spread, risk_free,
        )
        cip_results.append({
            "cip_basis": cip * 100,  # %表示
            "hedge_cost": fx_hc,
            "opt_h_fx": opt["opt_h_fx"],
            "opt_h_ir": opt["opt_h_ir"],
            "opt_sharpe": opt["opt_sharpe"],
        })
    results["cip_basis"] = cip_results

    # 2. デュレーション感応度
    duration_range = np.arange(4.0, 9.5, 0.5)
    dur_results = []
    for d in duration_range:
        opt = scipy_optimize(
            cov_matrix, e_i_spread, swap_rate, sofr, fx_hedge_cost_base,
            e_fx_return, d, d, risk_free,
        )
        dur_results.append({
            "duration": d,
            "opt_h_fx": opt["opt_h_fx"],
            "opt_h_ir": opt["opt_h_ir"],
            "opt_sharpe": opt["opt_sharpe"],
        })
    results["duration"] = dur_results

    # 3. ヘッジコスト水準感応度（FXヘッジコスト全体を変化）
    hedge_cost_range = np.arange(1.0, 6.5, 0.5)  # 1%〜6%
    hc_results = []
    for hc in hedge_cost_range:
        opt = scipy_optimize(
            cov_matrix, e_i_spread, swap_rate, sofr, hc, e_fx_return,
            D_base, D_spread, risk_free,
        )
        hc_results.append({
            "hedge_cost": hc,
            "opt_h_fx": opt["opt_h_fx"],
            "opt_h_ir": opt["opt_h_ir"],
            "opt_sharpe": opt["opt_sharpe"],
        })
    results["hedge_cost"] = hc_results

    return results
