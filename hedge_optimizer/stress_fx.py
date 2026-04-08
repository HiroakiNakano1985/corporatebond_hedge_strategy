"""為替ストレステスト

最適ヘッジ比率を固定した上で、ドル円が実際に変動した場合の
ポートフォリオリターンを計算する。
"""

from hedge_optimizer.analysis.covariance import get_latest_covariance
from hedge_optimizer.analysis.optimizer import scipy_optimize
from hedge_optimizer.analysis.returns import expected_return, portfolio_std
from hedge_optimizer.data.fetch_data import fetch_and_prepare_data


def run_stress_test():
    # データ取得
    weekly, changes, hist_stats = fetch_and_prepare_data()
    cov_matrix = get_latest_covariance(changes, method="rolling")

    # 市場データ
    e_i_spread = weekly["i_spread"].iloc[-1]
    swap_rate = weekly["swap_rate"].iloc[-1]
    sofr = weekly["sofr_90d"].iloc[-1]
    fx_hedge_cost = weekly["hedge_cost"].iloc[-1]
    e_fx_hist = hist_stats["e_fx_return"]

    # 過去平均ベースで最適化した比率を取得
    opt = scipy_optimize(
        cov_matrix, e_i_spread, swap_rate, sofr, fx_hedge_cost, e_fx_hist,
    )
    h_fx = opt["opt_h_fx"]
    h_ir = opt["opt_h_ir"]

    print("\n" + "=" * 80)
    print("為替ストレステスト")
    print("=" * 80)
    print(f"\n  固定ヘッジ比率: h_fx={h_fx:.2f}, h_ir={h_ir:.2f}")
    print(f"  (過去データ平均 E[USDJPY]={e_fx_hist:+.2f}% で最適化した比率)")
    print(f"\n  市場データ: I_spread={e_i_spread:.0f}bp, "
          f"Swap={swap_rate:.2f}%, SOFR={sofr:.2f}%, "
          f"FXヘッジコスト={fx_hedge_cost:.2f}%")

    # リターン分解を表示
    # r_p = I_spread/100 + (1-h_ir)*Swap + h_ir*SOFR + (1-h_fx)*Δ(USDJPY) - h_fx*FXcost
    carry_fixed = (1 - h_ir) * swap_rate
    carry_sofr = h_ir * sofr
    fx_cost = h_fx * fx_hedge_cost
    i_spread_pct = e_i_spread / 100.0
    fx_exposure = 1 - h_fx

    print(f"\n  リターン構成（為替変動以外の確定部分）:")
    print(f"    I_spread:              +{i_spread_pct:.2f}%")
    print(f"    固定金利キャリー:      +{carry_fixed:.2f}%  ((1-{h_ir:.2f})*{swap_rate:.2f}%)")
    print(f"    SOFR受取:              +{carry_sofr:.2f}%  ({h_ir:.2f}*{sofr:.2f}%)")
    print(f"    FXヘッジコスト:        -{fx_cost:.2f}%  ({h_fx:.2f}*{fx_hedge_cost:.2f}%)")
    base_return = i_spread_pct + carry_fixed + carry_sofr - fx_cost
    print(f"    小計（確定分）:        {base_return:+.2f}%")
    print(f"    為替エクスポージャー:  {fx_exposure:.2f}  (= 1 - h_fx)")

    # ストレスシナリオ
    fx_scenarios = [+10.0, +5.0, +3.0, 0.0, -3.0, -5.0, -10.0, -15.0, -20.0]

    sigma = portfolio_std(h_fx, h_ir, cov_matrix, annualize=True)

    print(f"\n  {'Δ(USDJPY)':>12s}  {'為替寄与':>10s}  {'トータルr_p':>12s}  {'Sharpe':>8s}")
    print("  " + "-" * 50)

    for fx in fx_scenarios:
        fx_contrib = fx_exposure * fx
        total = base_return + fx_contrib
        sharpe = total / sigma if sigma > 0 else 0.0

        marker = ""
        if fx == 0.0:
            marker = "  <-- 為替横ばい"
        elif abs(fx - e_fx_hist) < 0.5:
            marker = "  <-- 過去平均に近い"

        print(f"  {fx:>+11.1f}%  {fx_contrib:>+9.2f}%  {total:>+11.2f}%  "
              f"{sharpe:>8.2f}{marker}")

    # フルヘッジとの比較
    print(f"\n  --- 参考: フルヘッジ (h_fx=1.00) の場合 ---")
    full_hedge_r = i_spread_pct + sofr - fx_hedge_cost
    full_hedge_sigma = portfolio_std(1.0, h_ir, cov_matrix, annualize=True)
    full_hedge_sharpe = full_hedge_r / full_hedge_sigma if full_hedge_sigma > 0 else 0.0
    print(f"  リターン: {full_hedge_r:+.2f}%  σ: {full_hedge_sigma:.2f}%  "
          f"Sharpe: {full_hedge_sharpe:.2f}")
    print(f"  (為替変動の影響を一切受けない)")

    # 損益分岐点
    # base_return + fx_exposure * Δ(USDJPY) = full_hedge_r
    # → Δ(USDJPY) = (full_hedge_r - base_return) / fx_exposure
    if fx_exposure > 0.01:
        breakeven = (full_hedge_r - base_return) / fx_exposure
        print(f"\n  損益分岐点: Δ(USDJPY) = {breakeven:+.2f}%")
        print(f"  → ドル円が年率 {breakeven:+.1f}% 以上動けばフルヘッジより有利")

    print()


if __name__ == "__main__":
    run_stress_test()
