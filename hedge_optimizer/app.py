"""インタラクティブ・ヘッジシミュレーター（Streamlit）

スライドバーで h_fx / h_ir を操作し、
為替・金利の変化量を入力してリターンへの影響をリアルタイムに確認する。
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st

from hedge_optimizer.analysis.covariance import get_latest_covariance
from hedge_optimizer.analysis.returns import expected_return, portfolio_std
from hedge_optimizer.data.fetch_data import fetch_and_prepare_data


# ------------------------------------------------------------------
# データ取得（キャッシュ）
# ------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    weekly, changes, hist_stats = fetch_and_prepare_data()
    cov_rolling = get_latest_covariance(changes, method="rolling")
    return weekly, changes, hist_stats, cov_rolling


# ------------------------------------------------------------------
# メイン
# ------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="USD社債 ヘッジシミュレーター",
        page_icon="📊",
        layout="wide",
    )

    st.title("USD建IG社債 ヘッジ比率シミュレーター")
    st.caption("スライドバーでヘッジ比率を操作し、市場変動シナリオを入力してリターンへの影響を確認")

    # データロード
    with st.spinner("FREDからデータ取得中..."):
        weekly, changes, hist_stats, cov_matrix = load_data()

    # 最新市場データ
    latest = weekly.iloc[-1]
    e_i_spread = latest["i_spread"]
    swap_rate = latest["swap_rate"]
    sofr = latest["sofr_90d"]
    fx_hedge_cost = latest["hedge_cost"]
    usdjpy_now = latest["usdjpy"]
    e_fx_hist = hist_stats["e_fx_return"]

    # ==================================================================
    # サイドバー: パラメータ設定
    # ==================================================================
    st.sidebar.header("市場データ（最新値）")
    st.sidebar.metric("ドル円", f"{usdjpy_now:.2f}")
    st.sidebar.metric("I_spread", f"{e_i_spread:.0f} bp")
    st.sidebar.metric("スワップレート", f"{swap_rate:.2f}%")
    st.sidebar.metric("SOFR 90日", f"{sofr:.2f}%")
    st.sidebar.metric("FXヘッジコスト", f"{fx_hedge_cost:.2f}%")

    st.sidebar.divider()
    st.sidebar.header("デュレーション設定")
    D = st.sidebar.slider("修正デュレーション (年)", 3.0, 12.0, 6.5, 0.5)
    D_spread = st.sidebar.slider("スプレッドデュレーション (年)", 3.0, 12.0, D, 0.5)

    # ==================================================================
    # メインエリア: ヘッジ比率スライダー
    # ==================================================================
    col_slider1, col_slider2 = st.columns(2)

    with col_slider1:
        st.subheader("為替ヘッジ比率")
        h_fx = st.slider(
            "h_fx", 0.0, 1.0, 0.68, 0.01,
            help="0.0 = 為替フルオープン / 1.0 = 為替フルヘッジ",
        )
        st.caption(f"為替エクスポージャー: {(1-h_fx)*100:.0f}%")

    with col_slider2:
        st.subheader("金利ヘッジ比率")
        h_ir = st.slider(
            "h_ir", 0.0, 1.0, 1.00, 0.01,
            help="0.0 = 金利フルオープン / 1.0 = 金利フルヘッジ",
        )
        st.caption(f"金利エクスポージャー: {(1-h_ir)*100:.0f}%")

    st.divider()

    # ==================================================================
    # シナリオ入力
    # ==================================================================
    st.subheader("市場変動シナリオ（年率）")

    col_fx, col_ir, col_sp = st.columns(3)

    with col_fx:
        d_usdjpy = st.number_input(
            "ドル円変動 (%)",
            value=0.0, min_value=-50.0, max_value=50.0, step=1.0,
            help="正 = 円安 / 負 = 円高",
        )

    with col_ir:
        d_rate = st.number_input(
            "米金利変化 (%pt)",
            value=0.0, min_value=-3.0, max_value=3.0, step=0.25,
            help="正 = 金利上昇 / 負 = 金利低下",
        )

    with col_sp:
        d_spread = st.number_input(
            "I_spread変化 (bp)",
            value=0.0, min_value=-200.0, max_value=500.0, step=10.0,
            help="正 = スプレッド拡大 / 負 = スプレッド縮小",
        )

    # ==================================================================
    # リターン計算
    # ==================================================================

    # 確定キャリー部分
    carry_i_spread = e_i_spread / 100.0
    carry_fixed = (1 - h_ir) * swap_rate
    carry_sofr = h_ir * sofr
    cost_fx = h_fx * fx_hedge_cost
    base_return = carry_i_spread + carry_fixed + carry_sofr - cost_fx

    # 市場変動によるリターン
    fx_contrib = (1 - h_fx) * d_usdjpy
    ir_contrib = -(1 - h_ir) * D * d_rate
    spread_contrib = -D_spread * d_spread / 100.0

    total_return = base_return + fx_contrib + ir_contrib + spread_contrib

    # リスク指標
    sigma = portfolio_std(h_fx, h_ir, cov_matrix, D, D_spread, annualize=True)
    sharpe = total_return / sigma if sigma > 0.01 else 0.0

    # フルヘッジ比較
    full_hedge_return = carry_i_spread + sofr - fx_hedge_cost - D_spread * d_spread / 100.0
    full_hedge_sigma = portfolio_std(1.0, 1.0, cov_matrix, D, D_spread, annualize=True)

    # ==================================================================
    # 結果表示
    # ==================================================================
    st.divider()
    st.subheader("シミュレーション結果")

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)

    with col_r1:
        st.metric(
            "トータルリターン",
            f"{total_return:+.2f}%",
            delta=f"{total_return - full_hedge_return:+.2f}% vs フルヘッジ",
        )
    with col_r2:
        st.metric("年率σ", f"{sigma:.2f}%")
    with col_r3:
        st.metric("シャープレシオ", f"{sharpe:.2f}")
    with col_r4:
        st.metric(
            "フルヘッジリターン",
            f"{full_hedge_return:+.2f}%",
        )

    # ------------------------------------------------------------------
    # リターン分解ウォーターフォール
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("リターン分解")

    col_carry, col_market = st.columns(2)

    with col_carry:
        st.markdown("**確定キャリー**")
        data_carry = {
            "I_spread": carry_i_spread,
            f"固定金利 ((1-{h_ir:.2f})×{swap_rate:.1f}%)": carry_fixed,
            f"SOFR受取 ({h_ir:.2f}×{sofr:.1f}%)": carry_sofr,
            f"FXヘッジコスト ({h_fx:.2f}×{fx_hedge_cost:.1f}%)": -cost_fx,
        }
        for label, val in data_carry.items():
            color = "green" if val >= 0 else "red"
            st.markdown(f"- {label}: :{color}[**{val:+.2f}%**]")
        st.markdown(f"- **小計: {base_return:+.2f}%**")

    with col_market:
        st.markdown("**市場変動の寄与**")
        data_market = {
            f"為替 ((1-{h_fx:.2f})×{d_usdjpy:+.1f}%)": fx_contrib,
            f"金利 (-(1-{h_ir:.2f})×{D:.1f}×{d_rate:+.2f})": ir_contrib,
            f"スプレッド (-{D_spread:.1f}×{d_spread:+.0f}bp/100)": spread_contrib,
        }
        for label, val in data_market.items():
            color = "green" if val >= 0 else "red"
            st.markdown(f"- {label}: :{color}[**{val:+.2f}%**]")
        total_market = fx_contrib + ir_contrib + spread_contrib
        st.markdown(f"- **小計: {total_market:+.2f}%**")

    # ------------------------------------------------------------------
    # ストレステスト表
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("為替ストレステスト（金利・スプレッド変動は上記設定を維持）")

    fx_scenarios = [-20, -15, -10, -5, -3, 0, 3, 5, 10, 15, 20]
    stress_rows = []
    for fx in fx_scenarios:
        fx_c = (1 - h_fx) * fx
        ir_c = -(1 - h_ir) * D * d_rate
        sp_c = -D_spread * d_spread / 100.0
        total = base_return + fx_c + ir_c + sp_c
        sr = total / sigma if sigma > 0.01 else 0.0
        stress_rows.append({
            "Δ(USDJPY)": f"{fx:+d}%",
            "為替寄与": f"{fx_c:+.2f}%",
            "トータルr_p": f"{total:+.2f}%",
            "Sharpe": f"{sr:.2f}",
        })

    import pandas as pd
    stress_df = pd.DataFrame(stress_rows)
    st.dataframe(stress_df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # ドル円換算
    # ------------------------------------------------------------------
    if abs(d_usdjpy) > 0.01:
        implied_usdjpy = usdjpy_now * np.exp(d_usdjpy / 100)
        st.caption(f"参考: Δ(USDJPY)={d_usdjpy:+.1f}% → ドル円 {usdjpy_now:.2f} → {implied_usdjpy:.2f}")


if __name__ == "__main__":
    main()
