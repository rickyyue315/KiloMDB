import io
import math
from typing import List, Dict, Optional

import pandas as pd
import streamlit as st


# =============================
# App 基本設定
# =============================
st.set_page_config(
    page_title="Old Article RP & Ideal Stock Calculator",
    layout="wide",
)


# =============================
# 欄位與計算設定（可依 MDB 調整）
# =============================

# 基本必備欄位（用於識別與計算）
REQUIRED_COLUMNS = [
    "article_code",   # 貨品編號
    "article_name",   # 貨品名稱
    "current_stock",  # 現有庫存
]

# 可能會用到的欄位（若存在就用，沒有就用預設/計算）
OPTIONAL_COLUMNS = [
    "brand",
    "category",
    "sales_30d",
    "sales_90d",
    "avg_daily_sales",
    "lead_time_days",
    "safety_factor",
]


# =============================
# 欄位標題標準化：容忍不同命名
# =============================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    - 將欄位轉為 snake_case
    - 將常見別名對應到內部統一欄位
    方便用戶不用完全對齊 MDB 欄位名稱即可使用。
    """
    original_cols = list(df.columns)
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
    )

    alias_map: Dict[str, str] = {
        # article_code
        "item_code": "article_code",
        "item": "article_code",
        "sku": "article_code",
        "code": "article_code",
        # article_name
        "item_name": "article_name",
        "name": "article_name",
        "desc": "article_name",
        "description": "article_name",
        # current_stock
        "on_hand": "current_stock",
        "stock": "current_stock",
        "qty": "current_stock",
        "quantity": "current_stock",
        "inventory": "current_stock",
        # sales_30d / sales_90d
        "sales_30": "sales_30d",
        "sellout_30d": "sales_30d",
        "sales_last_30_days": "sales_30d",
        "sales_90": "sales_90d",
        "sellout_90d": "sales_90d",
        "sales_last_90_days": "sales_90d",
        # avg_daily_sales
        "ads": "avg_daily_sales",
        "avg_daily_sale": "avg_daily_sales",
        "average_daily_sales": "avg_daily_sales",
        # lead_time_days
        "lead_time": "lead_time_days",
        "lt_days": "lead_time_days",
        # safety_factor
        "safety": "safety_factor",
        "z_value": "safety_factor",
    }

    renamed = {}
    for col in df.columns:
        if col in alias_map:
            renamed[col] = alias_map[col]

    if renamed:
        df = df.rename(columns=renamed)

    df.attrs["original_columns"] = original_cols
    return df


def validate_required_columns(df: pd.DataFrame) -> List[str]:
    """檢查是否有缺少內部定義的必備欄位"""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


# =============================
# 計算邏輯（可對應 MDB 公式）
# =============================
def compute_metrics(
    df: pd.DataFrame,
    default_lead_time: float,
    default_safety_factor: float,
    default_min_order_qty: Optional[float],
) -> pd.DataFrame:
    """
    計算步驟 (可依 MDB 實際邏輯調整)：
    1. avg_daily_sales：
       - 若已有欄位則直接使用
       - 否則優先用 sales_90d / 90，其次 sales_30d / 30
    2. lead_time_days：
       - 若無欄位則使用 sidebar 預設值
    3. safety_factor：
       - 若無欄位則使用 sidebar 預設值
    4. safety_stock：
       - 範例公式: avg_daily_sales * safety_factor * (lead_time_days / 30)
    5. ideal_stock：
       - avg_daily_sales * lead_time_days + safety_stock
    6. rp_qty：
       - ideal_stock - current_stock，最少為 0，並可套用 MOQ 規則
    """
    df = df.copy()

    # 轉為數值類型
    for col in [
        "current_stock",
        "sales_30d",
        "sales_90d",
        "avg_daily_sales",
        "lead_time_days",
        "safety_factor",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 處理 avg_daily_sales
    if "avg_daily_sales" not in df.columns:
        df["avg_daily_sales"] = pd.NA

    mask_missing_ads = df["avg_daily_sales"].isna()

    if "sales_90d" in df.columns:
        df.loc[
            mask_missing_ads & df["sales_90d"].notna(), "avg_daily_sales"
        ] = df.loc[
            mask_missing_ads & df["sales_90d"].notna(), "sales_90d"
        ] / 90.0

    if "sales_30d" in df.columns:
        mask_missing_ads = df["avg_daily_sales"].isna()
        df.loc[
            mask_missing_ads & df["sales_30d"].notna(), "avg_daily_sales"
        ] = df.loc[
            mask_missing_ads & df["sales_30d"].notna(), "sales_30d"
        ] / 30.0

    df["avg_daily_sales"] = df["avg_daily_sales"].fillna(0)

    # lead_time_days
    if "lead_time_days" not in df.columns:
        df["lead_time_days"] = default_lead_time
    else:
        df["lead_time_days"] = df["lead_time_days"].fillna(default_lead_time)

    # safety_factor
    if "safety_factor" not in df.columns:
        df["safety_factor"] = default_safety_factor
    else:
        df["safety_factor"] = df["safety_factor"].fillna(default_safety_factor)

    # safety_stock
    df["safety_stock"] = (
        df["avg_daily_sales"] * df["safety_factor"] * (df["lead_time_days"] / 30.0)
    )

    # ideal_stock
    df["ideal_stock"] = (
        df["avg_daily_sales"] * df["lead_time_days"] + df["safety_stock"]
    )

    # rp_qty
    df["current_stock"] = df["current_stock"].fillna(0)
    df["rp_qty_raw"] = df["ideal_stock"] - df["current_stock"]
    df["rp_qty"] = df["rp_qty_raw"].apply(lambda x: max(math.floor(x), 0))

    # MOQ 邏輯
    if default_min_order_qty is not None and default_min_order_qty > 0:
        def apply_moq(qty: int) -> int:
            if qty <= 0:
                return 0
            return int(math.ceil(qty / default_min_order_qty) * default_min_order_qty)

        df["rp_qty"] = df["rp_qty"].apply(apply_moq)

    # 欄位順序整理
    preferred_order = [
        "article_code",
        "article_name",
        "brand",
        "category",
        "current_stock",
        "sales_30d",
        "sales_90d",
        "avg_daily_sales",
        "lead_time_days",
        "safety_factor",
        "safety_stock",
        "ideal_stock",
        "rp_qty",
    ]
    ordered_cols = [c for c in preferred_order if c in df.columns] + [
        c for c in df.columns if c not in preferred_order
    ]
    df = df[ordered_cols]

    return df


# =============================
# 匯出 Excel
# =============================
def to_excel_download(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="RP_Result")
    return output.getvalue()


# =============================
# Streamlit 畫面與流程設計
# =============================

st.title("Old Article RP & Ideal Stock Calculator")

st.markdown(
    """
此程式對應原本 Access/MDB「Upload Old Article RP Parameter (ideal stock added)」流程：

1. 上傳舊貨品 / 銷量 / 庫存 Excel 或 CSV
2. 系統自動標準化欄位名稱並檢查必要欄位
3. 根據參數 (Lead Time / Safety Factor / MOQ) 計算 RP 與理想庫存
4. 在畫面上檢視結果與 KPI
5. 下載計算結果 Excel

後續可依照你 MDB 裡實際公式微調本程式的計算邏輯。
"""
)

# -------- Sidebar：控制流程與參數 --------
with st.sidebar:
    st.header("Step 1 — 上傳資料與設定參數")

    uploaded_file = st.file_uploader(
        "上傳 Excel 或 CSV 檔",
        type=["xlsx", "xls", "csv"],
        help="內容需包含貨品編號、名稱與現有庫存，可另含銷量與其他欄位。",
    )

    st.markdown("---")
    st.subheader("預設計算參數")

    default_lead_time = st.number_input(
        "Default Lead Time (days)",
        min_value=1.0,
        max_value=365.0,
        value=30.0,
        step=1.0,
        help="若檔案中沒有 lead_time_days 欄位，將使用此值。",
    )

    default_safety_factor = st.number_input(
        "Default Safety Factor",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="安全係數，可對應原 MDB 設定服務水準。",
    )

    use_moq = st.checkbox("套用最小訂購量 MOQ", value=False)
    default_moq: Optional[float] = None
    if use_moq:
        default_moq = st.number_input(
            "Minimum Order Quantity (MOQ)",
            min_value=1.0,
            value=1.0,
            step=1.0,
            help="若有需求，可設定所有 RP 依 MOQ 進位。",
        )

    st.markdown("---")
    st.caption("上傳檔案後，系統會依以上規則計算 RP 與 Ideal Stock。")

# -------- Step 0：尚未上傳檔案 --------
if not uploaded_file:
    st.info("請先在左側上傳 Excel/CSV 檔以開始流程。")
else:
    # -------- Step 2：讀檔 & 預覽原始資料 --------
    try:
        if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
            df_raw = pd.read_excel(uploaded_file)
        else:
            df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"檔案讀取失敗：{e}")
        df_raw = None

    if df_raw is not None:
        st.subheader("Step 2 — 原始資料預覽")
        st.dataframe(df_raw.head(20), use_container_width=True)

        # -------- Step 3：欄位標準化與檢查 --------
        df_norm = normalize_columns(df_raw)
        missing = validate_required_columns(df_norm)

        with st.expander("欄位檢查與對應說明", expanded=bool(missing)):
            st.write("上傳檔案經標準化後的欄位：")
            st.write(list(df_norm.columns))
            if missing:
                st.error(
                    "缺少以下必備欄位： " + ", ".join(missing)
                )
                st.markdown(
                    """
請確認檔案中包含對應欄位，或調整程式中的 normalize_columns() 映射規則：

- article_code：貨品編號 (e.g. Item Code / SKU)
- article_name：貨品名稱
- current_stock：現有庫存 (e.g. On Hand / Stock / Qty)

修改後重新上傳即可繼續。
"""
                )

        if missing:
            st.stop()

        # -------- Step 4：計算 RP 與 Ideal Stock --------
        st.subheader("Step 3 — 計算結果")

        result_df = compute_metrics(
            df_norm,
            default_lead_time=default_lead_time,
            default_safety_factor=default_safety_factor,
            default_min_order_qty=default_moq,
        )

        # KPI 概覽
        total_skus = result_df["article_code"].nunique()
        total_rp_qty = result_df["rp_qty"].sum()
        skus_to_order = int((result_df["rp_qty"] > 0).sum())

        k1, k2, k3 = st.columns(3)
        k1.metric("Total SKUs", value=int(total_skus))
        k2.metric("SKUs with RP > 0", value=skus_to_order)
        k3.metric("Total Recommended Qty", value=int(total_rp_qty))

        # 篩選條件 (品牌 / 類別等)
        with st.expander("篩選條件 (可選)", expanded=False):
            filtered_df = result_df

            if "brand" in filtered_df.columns:
                brands = ["(All)"] + sorted(
                    [str(b) for b in filtered_df["brand"].dropna().unique()]
                )
                selected_brand = st.selectbox("Brand", brands, index=0)
                if selected_brand != "(All)":
                    filtered_df = filtered_df[
                        filtered_df["brand"].astype(str) == selected_brand
                    ]

            if "category" in filtered_df.columns:
                categories = ["(All)"] + sorted(
                    [str(c) for c in filtered_df["category"].dropna().unique()]
                )
                selected_cat = st.selectbox("Category", categories, index=0)
                if selected_cat != "(All)":
                    filtered_df = filtered_df[
                        filtered_df["category"].astype(str) == selected_cat
                    ]

        # 顯示最終表格
        st.dataframe(filtered_df, use_container_width=True)

        # -------- Step 5：匯出 Excel --------
        st.subheader("Step 4 — 匯出結果")

        excel_bytes = to_excel_download(filtered_df)
        st.download_button(
            label="下載 RP / Ideal Stock 結果 (Excel)",
            data=excel_bytes,
            file_name="rp_ideal_stock_result.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
        )

        st.success(
            "流程完成：已依目前參數計算 RP 與理想庫存，可下載結果檔案或調整參數後重新計算。"
        )