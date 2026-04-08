"""エントリーポイント

USD建IG社債の最適ヘッジ比率分析を実行する。
"""

import matplotlib.pyplot as plt

from hedge_optimizer.analysis.covariance import (
    get_latest_covariance,
    print_covariance_summary,
)
from hedge_optimizer.analysis.optimizer import (
    grid_search,
    scipy_optimize,
    sensitivity_analysis,
)
from hedge_optimizer.config import (
    DURATION,
    DURATION_SPREAD,
    E_FX_RETURN_OVERRIDE,
)
from hedge_optimizer.data.fetch_data import fetch_and_prepare_data
from hedge_optimizer.visualization.plots import (
    plot_3d_surface,
    plot_all_heatmaps,
    plot_sensitivity,
)


def main():
    # ========================================
    # 1. データ取得・前処理
    # ========================================
    weekly, changes, hist_stats = fetch_and_prepare_data()

    # ========================================
    # 2. 共分散行列推定
    # ========================================
    print("\n" + "=" * 50)
    print("共分散行列推定")
    print("=" * 50)

    cov_rolling = get_latest_covariance(changes, method="rolling")
    print("\n[ローリングウィンドウ]")
    print_covariance_summary(cov_rolling)

    cov_ewma = get_latest_covariance(changes, method="ewma")
    print("\n[EWMA]")
    print_covariance_summary(cov_ewma)

    # 分析にはローリング共分散を使用
    cov_matrix = cov_rolling

    # ========================================
    # 3. パラメータ設定
    # ========================================
    e_i_spread = weekly["i_spread"].iloc[-1]        # bp
    swap_rate = weekly["swap_rate"].iloc[-1]          # %
    sofr = weekly["sofr_90d"].iloc[-1]                # %
    fx_hedge_cost = weekly["hedge_cost"].iloc[-1]     # %
    sofr_90d = weekly["sofr_90d"].iloc[-1]
    japan_call = weekly["japan_call"].iloc[-1]

    # E[Δ(USDJPY)]: 上書き指定がなければ過去データのサンプル平均を使用
    if E_FX_RETURN_OVERRIDE is not None:
        e_fx_return = E_FX_RETURN_OVERRIDE
        fx_source = "上書き値"
    else:
        e_fx_return = hist_stats["e_fx_return"]
        fx_source = "過去データ平均"

    print(f"\n分析パラメータ:")
    print(f"  期待I_spread:      {e_i_spread:.0f} bp")
    print(f"  スワップレート:    {swap_rate:.2f}%")
    print(f"  SOFR (90日平均):   {sofr:.2f}%")
    print(f"  FXヘッジコスト:    {fx_hedge_cost:.2f}% (年率)")
    print(f"  E[Δ(USDJPY)]:      {e_fx_return:+.2f}% (年率, {fx_source})")
    print(f"  デュレーション:    {DURATION} 年")
    print(f"  スプレッドD:       {DURATION_SPREAD} 年")

    print(f"\n期待リターン分解 (h_fx=1, h_ir=1: フルヘッジ):")
    print(f"  I_spread:          +{e_i_spread/100:.2f}%")
    print(f"  SOFR (IR hedge):   +{sofr:.2f}%")
    print(f"  FXヘッジコスト:    -{fx_hedge_cost:.2f}%")
    er_full = e_i_spread / 100 + sofr - fx_hedge_cost
    print(f"  合計:              {er_full:.2f}%")

    print(f"\n期待リターン分解 (h_fx=0, h_ir=0: ノーヘッジ):")
    print(f"  I_spread:          +{e_i_spread/100:.2f}%")
    print(f"  Swap rate carry:   +{swap_rate:.2f}%")
    print(f"  E[Δ(USDJPY)]:      {e_fx_return:+.2f}%")
    er_none = e_i_spread / 100 + swap_rate + e_fx_return
    print(f"  合計:              {er_none:.2f}%")

    # ========================================
    # 4. Grid Search最適化
    # ========================================
    print("\n" + "=" * 50)
    print("Grid Search最適化")
    print("=" * 50)

    grid_result = grid_search(
        cov_matrix, e_i_spread, swap_rate, sofr, fx_hedge_cost, e_fx_return,
    )

    print(f"\n最適ヘッジ比率 (Grid Search):")
    print(f"  為替ヘッジ比率 (h_fx): {grid_result['opt_h_fx']:.2f}")
    print(f"  金利ヘッジ比率 (h_ir): {grid_result['opt_h_ir']:.2f}")
    print(f"  最大シャープレシオ:     {grid_result['opt_sharpe']:.4f}")
    print(f"  期待リターン:           {grid_result['opt_er']:.2f}%")
    print(f"  ポートフォリオ標準偏差: {grid_result['opt_std']:.2f}%")

    # ========================================
    # 5. SciPy精緻化最適化
    # ========================================
    print("\n" + "=" * 50)
    print("SciPy精緻化最適化")
    print("=" * 50)

    scipy_result = scipy_optimize(
        cov_matrix, e_i_spread, swap_rate, sofr, fx_hedge_cost, e_fx_return,
        x0=(grid_result["opt_h_fx"], grid_result["opt_h_ir"]),
    )

    print(f"\n最適ヘッジ比率 (SciPy):")
    print(f"  為替ヘッジ比率 (h_fx): {scipy_result['opt_h_fx']:.4f}")
    print(f"  金利ヘッジ比率 (h_ir): {scipy_result['opt_h_ir']:.4f}")
    print(f"  最大シャープレシオ:     {scipy_result['opt_sharpe']:.4f}")
    print(f"  期待リターン:           {scipy_result['opt_er']:.2f}%")
    print(f"  ポートフォリオ標準偏差: {scipy_result['opt_std']:.2f}%")
    print(f"  最適化収束: {scipy_result['result'].success}")

    # ========================================
    # 6. 感応度分析
    # ========================================
    print("\n" + "=" * 50)
    print("感応度分析")
    print("=" * 50)

    sens_results = sensitivity_analysis(
        cov_matrix, e_i_spread, swap_rate, sofr,
        sofr_90d=sofr_90d,
        japan_call=japan_call,
        fx_hedge_cost_base=fx_hedge_cost,
        e_fx_return=e_fx_return,
    )

    print("\nCIPベーシス感応度:")
    print(f"  {'CIP(%)':>8s}  {'h_fx':>6s}  {'h_ir':>6s}  {'Sharpe':>8s}")
    for r in sens_results["cip_basis"]:
        print(f"  {r['cip_basis']:8.1f}  {r['opt_h_fx']:6.2f}  "
              f"{r['opt_h_ir']:6.2f}  {r['opt_sharpe']:8.4f}")

    print("\nデュレーション感応度:")
    print(f"  {'D(年)':>8s}  {'h_fx':>6s}  {'h_ir':>6s}  {'Sharpe':>8s}")
    for r in sens_results["duration"]:
        print(f"  {r['duration']:8.1f}  {r['opt_h_fx']:6.2f}  "
              f"{r['opt_h_ir']:6.2f}  {r['opt_sharpe']:8.4f}")

    # ========================================
    # 7. 可視化
    # ========================================
    print("\n" + "=" * 50)
    print("プロット生成中...")
    print("=" * 50)

    fig_heatmaps = plot_all_heatmaps(grid_result)
    fig_heatmaps.savefig("heatmaps.png", bbox_inches="tight", dpi=150)
    print("  heatmaps.png を保存しました")

    fig_3d = plot_3d_surface(grid_result)
    fig_3d.savefig("surface_3d.png", bbox_inches="tight", dpi=150)
    print("  surface_3d.png を保存しました")

    fig_sens = plot_sensitivity(sens_results)
    fig_sens.savefig("sensitivity.png", bbox_inches="tight", dpi=150)
    print("  sensitivity.png を保存しました")

    plt.show()

    print("\n" + "=" * 50)
    print("分析完了")
    print("=" * 50)


if __name__ == "__main__":
    main()
