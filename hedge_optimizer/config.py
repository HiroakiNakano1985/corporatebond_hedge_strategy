"""パラメータ設定"""

# デュレーション
DURATION = 6.5            # 修正デュレーション（年）
DURATION_SPREAD = 6.5     # スプレッドデュレーション

# ヘッジコスト
CIP_BASIS = -0.004        # CIPベーシス（-0.4%）

# シャープレシオ
RISK_FREE_RATE = 0.0      # リスクフリーレート

# 為替期待変動の上書き値（年率%）
# None = 過去データのサンプル平均を使用（デフォルト）
# 0.0  = ランダムウォーク仮定
# 正の値 = 円安期待、負の値 = 円高期待
E_FX_RETURN_OVERRIDE = None

# 共分散推定
WINDOW_WEEKS = 52         # ローリングウィンドウ（週）
EWMA_LAMBDA = 0.94        # EWMA減衰係数

# Grid Search
H_GRID_SIZE = 50          # 分割数

# FREDシリーズコード
FRED_SERIES = {
    "oas":        "BAMLC0A0CM",       # ICE BofA IG社債OAS
    "ust_10y":    "DGS10",            # 米10年国債利回り
    "sofr":       "SOFR",             # SOFR翌日物
    "sofr_90d":   "SOFR90DAYAVG",     # SOFR 90日平均
    "usdjpy":     "DEXJPUS",          # ドル円
    "japan_call": "IRSTCI01JPM156N",  # 日本コールレート（TONA代替、月次）
}

# データ期間
DATA_YEARS = 5            # 過去5年分

# スワップスプレッド近似値（bp）
# 米国ではスワップスプレッドは現在マイナス
# I_spread ≈ OAS + Swap_Spread
SWAP_SPREAD_APPROX_BP = -20  # -20bp（近似値）
