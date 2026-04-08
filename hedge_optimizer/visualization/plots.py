"""可視化モジュール

4つのプロット:
  1. シャープレシオヒートマップ
  2. 期待リターンヒートマップ
  3. ポートフォリオ標準偏差ヒートマップ
  4. 3Dサーフェスプロット
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm


def _setup_style():
    """日本語対応のmatplotlibスタイル設定。"""
    plt.rcParams["font.family"] = ["MS Gothic", "Yu Gothic", "Meiryo", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120


def plot_heatmap(
    h_fx_vals: np.ndarray,
    h_ir_vals: np.ndarray,
    data_grid: np.ndarray,
    title: str,
    cbar_label: str,
    opt_h_fx: float,
    opt_h_ir: float,
    cmap: str = "RdYlGn",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """汎用ヒートマップ描画。"""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    im = ax.pcolormesh(
        h_fx_vals, h_ir_vals, data_grid,
        cmap=cmap, shading="auto",
    )
    plt.colorbar(im, ax=ax, label=cbar_label)

    # 最適点をマーク
    ax.plot(opt_h_fx, opt_h_ir, marker="*", color="black",
            markersize=15, markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate(
        f"({opt_h_fx:.2f}, {opt_h_ir:.2f})",
        (opt_h_fx, opt_h_ir),
        textcoords="offset points", xytext=(10, 10),
        fontsize=9, color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("h_fx (為替ヘッジ比率)")
    ax.set_ylabel("h_ir (金利ヘッジ比率)")
    ax.set_title(title)

    return ax


def plot_all_heatmaps(grid_result: dict) -> plt.Figure:
    """3つのヒートマップを一括描画する。"""
    _setup_style()

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    opt_hfx = grid_result["opt_h_fx"]
    opt_hir = grid_result["opt_h_ir"]
    h_fx_vals = grid_result["h_fx_vals"]
    h_ir_vals = grid_result["h_ir_vals"]

    # 1. シャープレシオ
    plot_heatmap(
        h_fx_vals, h_ir_vals, grid_result["sharpe_grid"],
        "シャープレシオ", "Sharpe Ratio",
        opt_hfx, opt_hir, cmap="RdYlGn", ax=axes[0],
    )

    # 2. 期待リターン
    plot_heatmap(
        h_fx_vals, h_ir_vals, grid_result["er_grid"],
        "期待リターン (%)", "E[r_p] (%)",
        opt_hfx, opt_hir, cmap="YlOrRd", ax=axes[1],
    )

    # 3. 標準偏差
    plot_heatmap(
        h_fx_vals, h_ir_vals, grid_result["std_grid"],
        "ポートフォリオ標準偏差 (%)", "σ_p (%)",
        opt_hfx, opt_hir, cmap="YlOrRd_r", ax=axes[2],
    )

    fig.suptitle(
        "USD建IG社債 最適ヘッジ比率分析",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_3d_surface(grid_result: dict) -> plt.Figure:
    """3Dサーフェスプロット。"""
    _setup_style()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    h_fx_grid = grid_result["h_fx_grid"]
    h_ir_grid = grid_result["h_ir_grid"]
    sharpe_grid = grid_result["sharpe_grid"]

    surf = ax.plot_surface(
        h_fx_grid, h_ir_grid, sharpe_grid,
        cmap=cm.RdYlGn,
        linewidth=0.1,
        antialiased=True,
        alpha=0.85,
    )

    # 最適点
    opt_hfx = grid_result["opt_h_fx"]
    opt_hir = grid_result["opt_h_ir"]
    opt_sr = grid_result["opt_sharpe"]
    ax.scatter(
        [opt_hfx], [opt_hir], [opt_sr],
        color="red", s=100, marker="*", zorder=5,
        label=f"最適点 ({opt_hfx:.2f}, {opt_hir:.2f})",
    )

    ax.set_xlabel("h_fx (為替ヘッジ比率)")
    ax.set_ylabel("h_ir (金利ヘッジ比率)")
    ax.set_zlabel("Sharpe Ratio")
    ax.set_title("シャープレシオ 3Dサーフェス")
    ax.legend()

    fig.colorbar(surf, shrink=0.5, aspect=10, label="Sharpe Ratio")
    return fig


def plot_sensitivity(sensitivity_results: dict) -> plt.Figure:
    """感応度分析プロット。"""
    _setup_style()

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 1. CIPベーシス感応度
    cip_data = sensitivity_results["cip_basis"]
    cip_vals = [d["cip_basis"] for d in cip_data]
    axes[0].plot(cip_vals, [d["opt_h_fx"] for d in cip_data], "o-", label="h_fx")
    axes[0].plot(cip_vals, [d["opt_h_ir"] for d in cip_data], "s-", label="h_ir")
    ax0_twin = axes[0].twinx()
    ax0_twin.plot(
        cip_vals, [d["opt_sharpe"] for d in cip_data],
        "^--", color="red", label="Sharpe",
    )
    axes[0].set_xlabel("CIPベーシス (%)")
    axes[0].set_ylabel("最適ヘッジ比率")
    ax0_twin.set_ylabel("Sharpe Ratio", color="red")
    axes[0].set_title("CIPベーシス感応度")
    axes[0].legend(loc="upper left")
    ax0_twin.legend(loc="upper right")

    # 2. デュレーション感応度
    dur_data = sensitivity_results["duration"]
    dur_vals = [d["duration"] for d in dur_data]
    axes[1].plot(dur_vals, [d["opt_h_fx"] for d in dur_data], "o-", label="h_fx")
    axes[1].plot(dur_vals, [d["opt_h_ir"] for d in dur_data], "s-", label="h_ir")
    ax1_twin = axes[1].twinx()
    ax1_twin.plot(
        dur_vals, [d["opt_sharpe"] for d in dur_data],
        "^--", color="red", label="Sharpe",
    )
    axes[1].set_xlabel("デュレーション (年)")
    axes[1].set_ylabel("最適ヘッジ比率")
    ax1_twin.set_ylabel("Sharpe Ratio", color="red")
    axes[1].set_title("デュレーション感応度")
    axes[1].legend(loc="upper left")
    ax1_twin.legend(loc="upper right")

    # 3. ヘッジコスト感応度
    hc_data = sensitivity_results["hedge_cost"]
    hc_vals = [d["hedge_cost"] for d in hc_data]
    axes[2].plot(hc_vals, [d["opt_h_fx"] for d in hc_data], "o-", label="h_fx")
    axes[2].plot(hc_vals, [d["opt_h_ir"] for d in hc_data], "s-", label="h_ir")
    ax2_twin = axes[2].twinx()
    ax2_twin.plot(
        hc_vals, [d["opt_sharpe"] for d in hc_data],
        "^--", color="red", label="Sharpe",
    )
    axes[2].set_xlabel("ヘッジコスト (%)")
    axes[2].set_ylabel("最適ヘッジ比率")
    ax2_twin.set_ylabel("Sharpe Ratio", color="red")
    axes[2].set_title("ヘッジコスト感応度")
    axes[2].legend(loc="upper left")
    ax2_twin.legend(loc="upper right")

    fig.suptitle("感応度分析", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig
