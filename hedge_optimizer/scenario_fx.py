"""ドル円期待変動シナリオ分析

E[Δ(USDJPY)] を変化させた場合の最適ヘッジ比率を比較する。
"""

import numpy as np

from hedge_optimizer.analysis.covariance import get_latest_covariance
from hedge_optimizer.analysis.optimizer import scipy_optimize
from hedge_optimizer.data.fetch_data import fetch_and_prepare_data


def run_fx_scenarios():
    # データ取得
    weekly, changes, hist_stats = fetch_and_prepare_data()

    # 共分散行列
    cov_matrix = get_latest_covariance(changes, method="rolling")

    # 市場データ
    e_i_spread = weekly["i_spread"].iloc[-1]
    swap_rate = weekly["swap_rate"].iloc[-1]
    sofr = weekly["sofr_90d"].iloc[-1]
    fx_hedge_cost = weekly["hedge_cost"].iloc[-1]

    # シナリオ定義
    scenarios = [
        ("過去平均",           hist_stats["e_fx_return"]),
        ("ランダムウォーク",   0.0),
        ("円高 -3%/年",       -3.0),
        ("円高 -5%/年",       -5.0),
        ("円高 -10%/年",     -10.0),
    ]

    # ヘッダー
    print("\n" + "=" * 80)
    print("ドル円シナリオ分析")
    print("=" * 80)
    print(f"\n  市場データ: I_spread={e_i_spread:.0f}bp, "
          f"Swap={swap_rate:.2f}%, SOFR={sofr:.2f}%, "
          f"FXヘッジコスト={fx_hedge_cost:.2f}%")
    print()
    print(f"  {'シナリオ':<20s}  {'E[USDJPY]':>10s}  {'h_fx':>6s}  "
          f"{'h_ir':>6s}  {'Sharpe':>8s}  {'E[r_p]':>8s}  {'σ_p':>8s}")
    print("  " + "-" * 74)

    for name, e_fx in scenarios:
        result = scipy_optimize(
            cov_matrix, e_i_spread, swap_rate, sofr, fx_hedge_cost, e_fx,
        )
        print(f"  {name:<20s}  {e_fx:>+9.2f}%  {result['opt_h_fx']:>6.2f}  "
              f"{result['opt_h_ir']:>6.2f}  {result['opt_sharpe']:>8.4f}  "
              f"{result['opt_er']:>+7.2f}%  {result['opt_std']:>7.2f}%")

    print()


if __name__ == "__main__":
    run_fx_scenarios()
