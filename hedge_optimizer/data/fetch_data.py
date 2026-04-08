"""FREDからのデータ取得モジュール

週次データ（過去5年）を取得し、前処理を行う。
- OAS, 米10年国債, SOFR, SOFR90日平均, ドル円, 日本コールレート
- 月次データ（日本コールレート）は週次に線形補間
- 欠損値は前値補完（ffill）
- I_spread = OAS + Swap_Spread の変換
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from dotenv import load_dotenv

from hedge_optimizer.config import (
    DATA_YEARS,
    FRED_SERIES,
    SWAP_SPREAD_APPROX_BP,
)

# プロジェクトルートの .env を読み込む
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")


def _get_fred_api_key() -> str:
    """環境変数またはStreamlit SecretsからFRED APIキーを取得する。"""
    # 1. 環境変数から取得
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key
    # 2. Streamlit Secrets から取得（Streamlit Cloud用）
    try:
        import streamlit as st
        key = st.secrets.get("FRED_API_KEY")
        if key:
            return key
    except Exception:
        pass
    raise EnvironmentError(
        "FRED_API_KEY が設定されていません。"
        ".env ファイル、環境変数、またはStreamlit Secretsで設定してください。"
    )


def _date_range(years: int = DATA_YEARS) -> tuple[datetime, datetime]:
    """取得期間（過去N年〜今日）を返す。"""
    end = datetime.today()
    start = end - timedelta(days=years * 365)
    return start, end


def fetch_single_series(series_id: str, start: datetime, end: datetime) -> pd.Series:
    """FREDから単一系列を取得する。"""
    api_key = _get_fred_api_key()
    df = web.DataReader(series_id, "fred", start, end, api_key=api_key)
    return df.iloc[:, 0]


def fetch_all_series() -> pd.DataFrame:
    """全FREDシリーズを取得し、1つのDataFrameにまとめる。

    Returns:
        columns: oas, ust_10y, sofr, sofr_90d, usdjpy, japan_call
        index: DatetimeIndex（日次、欠損あり）
    """
    start, end = _date_range()
    frames = {}

    for name, series_id in FRED_SERIES.items():
        print(f"  取得中: {name} ({series_id}) ...")
        try:
            frames[name] = fetch_single_series(series_id, start, end)
        except Exception as e:
            print(f"  警告: {name} の取得に失敗しました: {e}")
            frames[name] = pd.Series(dtype=float)

    df = pd.DataFrame(frames)
    df.index.name = "date"
    return df


def interpolate_monthly_to_weekly(series: pd.Series) -> pd.Series:
    """月次データを週次（金曜日）に線形補間する。"""
    if series.empty:
        return series
    # 日次にリサンプル → 線形補間 → 週次（金曜日）にリサンプル
    daily = series.resample("D").interpolate(method="linear")
    weekly = daily.resample("W-FRI").last()
    return weekly


def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """日次データを週次（金曜日）にリサンプルする。

    - 日本コールレート（月次）は線形補間後に週次化
    - その他は週次の最終値を採用
    """
    # 日本コールレートは月次なので別処理
    japan_call = interpolate_monthly_to_weekly(df["japan_call"])

    # その他は週次にリサンプル（最終値）
    weekly = df.drop(columns=["japan_call"]).resample("W-FRI").last()

    # 日本コールレートを結合
    weekly["japan_call"] = japan_call

    return weekly


def forward_fill(df: pd.DataFrame) -> pd.DataFrame:
    """欠損値を前値補完する。"""
    return df.ffill()


def compute_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """派生フィールドを計算する。

    - swap_spread: スワップスプレッド近似値（定数）
    - swap_rate: スワップレート近似 = UST10Y + swap_spread
    - i_spread: I_spread ≈ OAS + Swap_Spread
    - hedge_cost: 為替ヘッジコスト = SOFR_90d - japan_call + CIP_basis
      （年率%で計算）
    """
    from hedge_optimizer.config import CIP_BASIS

    df = df.copy()

    # スワップスプレッド（bp → %）
    swap_spread_pct = SWAP_SPREAD_APPROX_BP / 100.0

    # スワップレート近似
    df["swap_rate"] = df["ust_10y"] + swap_spread_pct

    # I_spread = OAS + Swap_Spread
    # OASは対国債スプレッドなので、I_spread（対スワップ）への変換
    # I_spread = OAS - Swap_Spread（Swap_Spreadがマイナスなので I_spread > OAS）
    # 注: OASはbp単位、swap_spreadもbp単位で計算
    df["i_spread"] = df["oas"] - SWAP_SPREAD_APPROX_BP  # bp単位

    # 為替ヘッジコスト（年率%）
    # SOFR_90d, japan_call は%単位
    # CIP_BASISは小数（例: -0.004 = -0.4%）なので%に変換
    df["hedge_cost"] = df["sofr_90d"] - df["japan_call"] + CIP_BASIS * 100

    return df


def compute_weekly_changes(df: pd.DataFrame) -> pd.DataFrame:
    """週次変化量を計算する。

    Returns:
        DataFrame with columns:
        - d_rate_usd: 米金利変化（%ポイント）
        - d_usdjpy: ドル円変化（対数リターン）
        - d_i_spread: I_spread変化（bp）
    """
    changes = pd.DataFrame(index=df.index)

    # 米金利変化（%ポイント）
    changes["d_rate_usd"] = df["ust_10y"].diff()

    # ドル円変化（対数リターン）
    changes["d_usdjpy"] = np.log(df["usdjpy"] / df["usdjpy"].shift(1))

    # I_spread変化（bp）
    changes["d_i_spread"] = df["i_spread"].diff()

    # 最初の行はNaNになるので削除
    changes = changes.dropna()

    return changes


def compute_historical_stats(changes: pd.DataFrame) -> dict:
    """週次変化量からファクターの統計量を計算する。

    Returns:
        dict with keys:
        - mean_weekly: 各ファクターの週次平均
        - mean_annual: 各ファクターの年率平均（×52）
        - std_weekly: 各ファクターの週次標準偏差
        - std_annual: 年率標準偏差（×√52）
        - e_fx_return: E[Δ(USDJPY)] の年率値（%換算）
    """
    cols = ["d_rate_usd", "d_usdjpy", "d_i_spread"]

    mean_w = changes[cols].mean()
    std_w = changes[cols].std()

    # 年率換算
    mean_a = mean_w * 52
    std_a = std_w * np.sqrt(52)

    # E[Δ(USDJPY)] は対数リターンなので年率化して%変換（×100）
    e_fx_return = mean_a["d_usdjpy"] * 100

    return {
        "mean_weekly": mean_w,
        "mean_annual": mean_a,
        "std_weekly": std_w,
        "std_annual": std_a,
        "e_fx_return": e_fx_return,
    }


def fetch_and_prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """データ取得から前処理まで一括実行する。

    Returns:
        (weekly_data, weekly_changes, hist_stats)
        - weekly_data: 週次データ（レベル値・派生フィールド含む）
        - weekly_changes: 週次変化量（3ファクター）
        - hist_stats: ファクターの統計量（平均・標準偏差等）
    """
    print("=" * 50)
    print("FREDからデータを取得中...")
    print("=" * 50)

    # 1. 全系列取得
    raw = fetch_all_series()
    print(f"\n取得完了: {len(raw)} 日分のデータ")
    print(f"期間: {raw.index.min()} 〜 {raw.index.max()}")

    # 2. 週次リサンプル
    print("\n週次リサンプル中...")
    weekly = resample_to_weekly(raw)

    # 3. 前値補完
    weekly = forward_fill(weekly)
    print(f"週次データ: {len(weekly)} 週分")

    # 4. 派生フィールド計算
    print("派生フィールド計算中...")
    weekly = compute_derived_fields(weekly)

    # 5. 週次変化量
    print("週次変化量計算中...")
    changes = compute_weekly_changes(weekly)
    print(f"変化量データ: {len(changes)} 週分")

    # 6. ファクター統計量
    print("ファクター統計量計算中...")
    hist_stats = compute_historical_stats(changes)

    # サマリー表示
    print("\n" + "=" * 50)
    print("データサマリー")
    print("=" * 50)
    print(f"\n最新データ ({weekly.index[-1].strftime('%Y-%m-%d')}):")
    print(f"  OAS:           {weekly['oas'].iloc[-1]:.0f} bp")
    print(f"  I_spread:      {weekly['i_spread'].iloc[-1]:.0f} bp")
    print(f"  米10年金利:    {weekly['ust_10y'].iloc[-1]:.2f}%")
    print(f"  SOFR 90日:     {weekly['sofr_90d'].iloc[-1]:.2f}%")
    print(f"  日本コールレート: {weekly['japan_call'].iloc[-1]:.2f}%")
    print(f"  ドル円:        {weekly['usdjpy'].iloc[-1]:.2f}")
    print(f"  ヘッジコスト:  {weekly['hedge_cost'].iloc[-1]:.2f}% (年率)")

    print(f"\nファクター平均（年率）:")
    print(f"  ΔR_USD:        {hist_stats['mean_annual']['d_rate_usd']:+.4f} %pt")
    print(f"  Δ(USDJPY):     {hist_stats['e_fx_return']:+.2f}%")
    print(f"  ΔI_spread:     {hist_stats['mean_annual']['d_i_spread']:+.2f} bp")

    print(f"\nファクター標準偏差（年率）:")
    print(f"  ΔR_USD:        {hist_stats['std_annual']['d_rate_usd']:.4f} %pt")
    print(f"  Δ(USDJPY):     {hist_stats['std_annual']['d_usdjpy']*100:.2f}%")
    print(f"  ΔI_spread:     {hist_stats['std_annual']['d_i_spread']:.2f} bp")

    return weekly, changes, hist_stats


if __name__ == "__main__":
    weekly, changes, hist_stats = fetch_and_prepare_data()
    print("\n週次データ（最新5行）:")
    print(weekly.tail())
    print("\n週次変化量（最新5行）:")
    print(changes.tail())
