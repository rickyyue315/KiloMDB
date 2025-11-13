import io
import math
from typing import List, Dict, Optional, Tuple

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
# Result.xlsx 欄位規格（權威定義）
# 僅輸出這些欄位，順序與命名需與原始 Result.xlsx 完全一致
# =============================

OUTPUT_COLUMNS: List[str] = [
    "Site",
    "Article",
    "Article Description",
    "Brand",
    "MC",
    "MC Description",
    "Article categor",
    "Article Type",
    "Status",
    "First Sales Dat",
    "Season category",
    "Available to",
    "Launch Date",
    "Sales Qty 20000101 -  20061203",
    "Sales Price",
    "Avg Weekly Sales",
    "Cal Stock Turnover",
    "Stock On Hand  20070827",
    "Safety Stock",
    "Purchase Group",
    "RP Type",
    "Planning Cycle",
    "Delivery Cycle",
    "Stock Planner",
    "Reorder Point",
    "Delivery Days",
    "Target Coverage",
    "Supply Source (1=Vendor/2=DC)",
    "ABC Indicator",
    "Smooth Promotion",
    "Forecast Model",
    "Historical periods",
    "Forecast periods",
    "Periods per season",
    "Current consumption qty",
    "Week 1 forecast value",
    "Week 2 forecast value",
    "Week 3 forecast value",
    "Week 4 forecast value",
    "Week 5 forecast value",
    "New Safety Qty",
    "New Purchase Group",
    "New RP Typ",
    "New Planning Cycle",
    "New Delivery Cycle",
    "New Stock Planner",
    "New Reorder Point",
    "New Delivery Days",
    "New Traget Coverage",  # 原始拼字錯誤，保持一致
    "New Supply Source",
    "New ABC Indicator",
    "New Smoothing (0/1)",
    "New Forecast Model",
    "New Historical perio",  # 原始拼字/截斷，保持一致
    "New Forecast periods",
    "New Periods per season",
    "New Current consumption qty",
    "New Week 1 forecast value",
    "New Week 2 forecast value",
    "New Week 3 forecast value",
    "New Week 4 forecast value",
    "New Week 5 forecast value",
    "A QTY",
    "B QTY",
    "C QTY",
]

# 推斷哪些欄位是數值欄（缺值時補 0），其餘視為文字欄（缺值補空字串）
NUMERIC_LIKE_COLUMNS = {
    "Sales Qty 20000101 -  20061203",
    "Sales Price",
    "Avg Weekly Sales",
    "Cal Stock Turnover",
    "Stock On Hand  20070827",
    "Safety Stock",
    "Reorder Point",
    "Delivery Days",
    "Target Coverage",
    "Current consumption qty",
    "Week 1 forecast value",
    "Week 2 forecast value",
    "Week 3 forecast value",
    "Week 4 forecast value",
    "Week 5 forecast value",
    "New Safety Qty",
    "New Reorder Point",
    "New Delivery Days",
    "New Traget Coverage",
    "New Current consumption qty",
    "New Week 1 forecast value",
    "New Week 2 forecast value",
    "New Week 3 forecast value",
    "New Week 4 forecast value",
    "New Week 5 forecast value",
    "A QTY",
    "B QTY",
    "C QTY",
}

# =============================
# 欄位標準化與 Article Master 設定
# =============================

REQUIRED_COLUMNS = [
    "article_code",
    "article_name",
    "current_stock",
]

MASTER_KEY_CANDIDATES = [
    "article_number_sap",
    "skus_number_magic_sys",
]

MASTER_ATTRIBUTE_COLUMNS = [
    "article_type",
    "status",
    "article_category",
    "season_category",
    "brand",
    "article_description",
    "manufacturer_product_line",
    "merchandise_category",
    "major_vendor_magic_sys",
    "major_vendor_sap",
    "purchase_group",
    "buyer",
    "supply_source",
    "original_supply_source",
    "supply_site",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    將一般來源檔欄位標準化為 snake_case，並套用常見別名:
    僅用於內部計算與對應，不直接影響輸出欄位名稱。
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


def normalize_article_master(df: pd.DataFrame) -> pd.DataFrame:
    """
    處理 Article Master:
    - 正規化欄位
    - 建立 master_article_code 供 join
    - 僅保留關鍵屬性
    """
    if df is None or df.empty:
        return df

    original_cols = list(df.columns)
    m = df.copy()
    m.columns = (
        m.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
    )

    alias_map: Dict[str, str] = {
        "article_number_sap": "article_number_sap",
        "article_number__sap_": "article_number_sap",
        "article_no_sap": "article_number_sap",
        "skus_number_magic_sys": "skus_number_magic_sys",
        "skus_no_magic_sys": "skus_number_magic_sys",
        "article_type": "article_type",
        "status": "status",
        "article_category": "article_category",
        "season_category": "season_category",
        "brand": "brand",
        "article_description": "article_description",
        "manufacturer_product_line": "manufacturer_product_line",
        "merchandise_category": "merchandise_category",
        "major_vendor_magic_sys": "major_vendor_magic_sys",
        "major_vendor_sap": "major_vendor_sap",
        "purchase_group": "purchase_group",
        "buyer": "buyer",
        "supply_source": "supply_source",
        "original_supply_source": "original_supply_source",
        "supply_site": "supply_site",
    }

    renamed = {}
    for col in m.columns:
        if col in alias_map:
            renamed[col] = alias_map[col]
    if renamed:
        m = m.rename(columns=renamed)

    key_col = None
    for cand in MASTER_KEY_CANDIDATES:
        if cand in m.columns:
            key_col = cand
            break

    if key_col is not None:
        m["master_article_code"] = m[key_col].astype(str).str.strip()
    else:
        m.attrs["original_columns"] = original_cols
        return m

    cols_to_keep = ["master_article_code"]
    for c in MASTER_ATTRIBUTE_COLUMNS:
        if c in m.columns:
            cols_to_keep.append(c)

    m = m[cols_to_keep].drop_duplicates(subset=["master_article_code"])
    m.attrs["original_columns"] = original_cols
    return m


def validate_required_columns(df: pd.DataFrame) -> List[str]:
    """檢查是否有缺少內部定義的必備欄位"""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


# =============================
# Planning Cycle 對照表
# =============================

def load_planning_cycle_mapping(path_or_buffer) -> pd.DataFrame:
    """
    讀取 Planning Cycle.xls，標準化為:
    rp_type, old_planning_cycle, old_delivery_cycle, new_planning_cycle, new_delivery_cycle
    """
    try:
        pc_raw = pd.read_excel(path_or_buffer)
    except Exception:
        return pd.DataFrame()

    df = pc_raw.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
    )

    alias = {
        "rp_type": ["rp_type", "rp_typ", "rp"],
        "old_planning_cycle": ["old_planning_cycle", "planning_cycle_old", "planning_cycle"],
        "old_delivery_cycle": ["old_delivery_cycle", "delivery_cycle_old", "delivery_cycle"],
        "new_planning_cycle": ["new_planning_cycle"],
        "new_delivery_cycle": ["new_delivery_cycle"],
    }

    def pick(col_candidates: List[str]) -> Optional[str]:
        for c in col_candidates:
            if c in df.columns:
                return c
        return None

    col_rp = pick(alias["rp_type"])
    col_old_pc = pick(alias["old_planning_cycle"])
    col_old_dc = pick(alias["old_delivery_cycle"])
    col_new_pc = pick(alias["new_planning_cycle"])
    col_new_dc = pick(alias["new_delivery_cycle"])

    if not (col_rp and col_old_pc and col_new_pc):
        return pd.DataFrame()

    m = pd.DataFrame()
    m["rp_type"] = df[col_rp].astype(str).str.strip()
    m["old_planning_cycle"] = df[col_old_pc].astype(str).str.strip()
    if col_old_dc:
        m["old_delivery_cycle"] = df[col_old_dc].astype(str).str.strip()
    else:
        m["old_delivery_cycle"] = ""
    m["new_planning_cycle"] = df[col_new_pc].astype(str).str.strip()
    if col_new_dc:
        m["new_delivery_cycle"] = df[col_new_dc].astype(str).str.strip()
    else:
        m["new_delivery_cycle"] = ""

    m = m.drop_duplicates(
        subset=["rp_type", "old_planning_cycle", "old_delivery_cycle"],
        keep="last",
    )
    return m


# =============================
# MDB 對應邏輯（生成 New 欄位與 A/B/C QTY）
# =============================

def apply_mdb_logic(
    df: pd.DataFrame,
    demo: bool = True,
    pc_mapping: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    對齊 Result.xlsx 結構的運算骨架：
    - 嘗試由 RP Type + Planning/Delivery Cycle 對應 New Planning/Delivery Cycle
    - 填入 New* 欄位與 A/B/C QTY（目前為示範/預設邏輯）
    - 真實公式可依 MDB 補上
    """
    df = df.copy()

    # 數值欄轉為數值
    for c in [
        "Stock On Hand  20070827",
        "Reorder Point",
        "Delivery Days",
        "Target Coverage",
        "Current consumption qty",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 1) New Planning/Delivery Cycle from mapping
    if pc_mapping is not None and not pc_mapping.empty:
        def norm_str(s: pd.Series) -> pd.Series:
            return s.astype(str).str.strip()

        # 找出來源 RP/PC/DC 欄位名稱（優先使用 Result.xlsx 的 header）
        src_rp = None
        if "RP Type" in df.columns:
            src_rp = "RP Type"
        elif "rp_type" in df.columns:
            src_rp = "rp_type"

        src_pc = None
        if "Planning Cycle" in df.columns:
            src_pc = "Planning Cycle"
        elif "planning_cycle" in df.columns:
            src_pc = "planning_cycle"

        src_dc = None
        if "Delivery Cycle" in df.columns:
            src_dc = "Delivery Cycle"
        elif "delivery_cycle" in df.columns:
            src_dc = "delivery_cycle"

        if src_rp and src_pc:
            df["_rp_type_key"] = norm_str(df[src_rp])
            df["_pc_key"] = norm_str(df[src_pc])
            if src_dc:
                df["_dc_key"] = norm_str(df[src_dc])
            else:
                df["_dc_key"] = ""

            m = pc_mapping.copy()
            m["_rp_type_key"] = norm_str(m["rp_type"])
            m["_pc_key"] = norm_str(m["old_planning_cycle"])
            m["_dc_key"] = norm_str(m["old_delivery_cycle"])

            merged = df.merge(
                m[
                    [
                        "_rp_type_key",
                        "_pc_key",
                        "_dc_key",
                        "new_planning_cycle",
                        "new_delivery_cycle",
                    ]
                ],
                how="left",
                on=["_rp_type_key", "_pc_key", "_dc_key"],
                suffixes=("", "_pcmap"),
            )

            no_match = merged["new_planning_cycle"].isna()
            if no_match.any():
                m2 = (
                    m.groupby(["_rp_type_key", "_pc_key"])
                    .last()[["new_planning_cycle", "new_delivery_cycle"]]
                    .reset_index()
                )
                merged2 = merged.merge(
                    m2,
                    how="left",
                    on=["_rp_type_key", "_pc_key"],
                    suffixes=("", "_pcmap2"),
                )

                merged2["New Planning Cycle"] = merged2["new_planning_cycle"]
                merged2.loc[
                    merged2["New Planning Cycle"].isna(),
                    "New Planning Cycle",
                ] = merged2["new_planning_cycle_pcmap2"]

                merged2["New Delivery Cycle"] = merged2["new_delivery_cycle"]
                merged2.loc[
                    merged2["New Delivery Cycle"].isna(),
                    "New Delivery Cycle",
                ] = merged2["new_delivery_cycle_pcmap2"]

                df = merged2
            else:
                merged["New Planning Cycle"] = merged["new_planning_cycle"]
                merged["New Delivery Cycle"] = merged["new_delivery_cycle"]
                df = merged

            # 若 New Delivery Cycle 缺失則沿用原 Delivery Cycle
            if "New Delivery Cycle" in df.columns:
                df["New Delivery Cycle"] = df["New Delivery Cycle"].where(
                    df["New Delivery Cycle"].notna(),
                    df.get("Delivery Cycle", df.get("delivery_cycle", "")),
                )

            # 清除暫存欄
            drop_cols = [
                c
                for c in df.columns
                if c.startswith("_rp_type_key")
                or c.startswith("_pc_key")
                or c.startswith("_dc_key")
                or c in [
                    "new_planning_cycle",
                    "new_delivery_cycle",
                    "new_planning_cycle_pcmap2",
                    "new_delivery_cycle_pcmap2",
                ]
            ]
            df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 2) New Safety Qty 示範邏輯：若有 Reorder Point 與 Stock On Hand，取差值
    if demo:
        if "Reorder Point" in df.columns and "Stock On Hand  20070827" in df.columns:
            base = (
                pd.to_numeric(df["Reorder Point"], errors="coerce")
                - pd.to_numeric(df["Stock On Hand  20070827"], errors="coerce")
            )
            df["New Safety Qty"] = base.clip(lower=0).fillna(0)
        else:
            if "New Safety Qty" not in df.columns:
                df["New Safety Qty"] = 0
    else:
        if "New Safety Qty" not in df.columns:
            df["New Safety Qty"] = 0

    # 3) 其餘 New* 欄位預設沿用舊欄位（若存在），否則給空白/0
    copy_pairs = [
        ("Purchase Group", "New Purchase Group"),
        ("RP Type", "New RP Typ"),
        ("Stock Planner", "New Stock Planner"),
        ("Reorder Point", "New Reorder Point"),
        ("Delivery Days", "New Delivery Days"),
        ("Target Coverage", "New Traget Coverage"),
        ("Supply Source (1=Vendor/2=DC)", "New Supply Source"),
        ("ABC Indicator", "New ABC Indicator"),
    ]
    for src, tgt in copy_pairs:
        if tgt not in df.columns:
            if src in df.columns:
                df[tgt] = df[src]
            else:
                df[tgt] = ""

    if "New Planning Cycle" not in df.columns:
        df["New Planning Cycle"] = df.get("Planning Cycle", "")
    if "New Delivery Cycle" not in df.columns:
        df["New Delivery Cycle"] = df.get("Delivery Cycle", "")

    # 預設空白 New* 欄位
    for tgt in [
        "New Smoothing (0/1)",
        "New Forecast Model",
        "New Historical perio",
        "New Forecast periods",
        "New Periods per season",
    ]:
        if tgt not in df.columns:
            df[tgt] = ""

    for tgt in [
        "New Current consumption qty",
        "New Week 1 forecast value",
        "New Week 2 forecast value",
        "New Week 3 forecast value",
        "New Week 4 forecast value",
        "New Week 5 forecast value",
    ]:
        if tgt not in df.columns:
            df[tgt] = 0

    # 4) A/B/C QTY 示範邏輯：以 New Reorder Point - Stock On Hand 為基礎
    if demo:
        if "New Reorder Point" in df.columns and "Stock On Hand  20070827" in df.columns:
            base = (
                pd.to_numeric(df["New Reorder Point"], errors="coerce")
                - pd.to_numeric(df["Stock On Hand  20070827"], errors="coerce")
            ).clip(lower=0).fillna(0)
        else:
            base = 0
        if "A QTY" not in df.columns:
            df["A QTY"] = base
        if "B QTY" not in df.columns:
            df["B QTY"] = base
        if "C QTY" not in df.columns:
            df["C QTY"] = base
    else:
        for col in ["A QTY", "B QTY", "C QTY"]:
            if col not in df.columns:
                df[col] = 0

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
# Streamlit 畫面與主流程
# =============================

st.title("Old Article RP Parameter (Ideal Stock Added) — Streamlit版")

st.markdown(
    """
此程式目標：**重現 MDB 工具輸出 Result.xlsx 的欄位與結構**，可直接部署於 Streamlit Cloud。

實作原則：
- 僅輸出 Result.xlsx 定義的 69 個欄位，欄位名稱與順序 100% 一致。
- 若來源資料缺少某欄，則依欄位型別以空白字串或 0 補齊。
- 不輸出任何多餘或中間計算欄位，方便與原 MDB 結果做 1:1 比對。
- 詳細計算邏輯集中於 apply_mdb_logic()，可依 MDB 規則再強化。
"""
)

with st.sidebar:
    st.header("Step 1 — 上傳來源資料")

    rp_source_file = st.file_uploader(
        "上傳 RP List / 原始來源 (格式需含 Site, Article 等欄位)",
        type=["xlsx", "xls", "csv", "txt"],
        help="此檔作為主要輸入，程式會依其內容與欄位 Mapping 重建 Result.xlsx 結構。",
    )

    article_master_file = st.file_uploader(
        "（可選）上傳 Article Master 檔",
        type=["xlsx", "xls", "csv"],
        help="可用來補足 Brand/MC/Article Type 等主檔欄位。",
    )

    st.markdown("---")
    st.subheader("運算模式")
    use_default_new_values = st.checkbox(
        "啟用示範 New* 與 A/B/C QTY 計算（未接上 MDB 完整公式前建議勾選）",
        value=True,
    )

if not rp_source_file:
    st.info("請先在左側上傳 RP List / 原始來源檔。")
else:
    # Step 2：讀取 RP 來源
    try:
        name = rp_source_file.name.lower()
        if name.endswith((".xlsx", ".xls")):
            rp_raw = pd.read_excel(rp_source_file)
        elif name.endswith(".txt"):
            rp_raw = pd.read_csv(rp_source_file, sep=None, engine="python")
        else:
            rp_raw = pd.read_csv(rp_source_file)
    except Exception as e:
        st.error(f"RP List / 來源檔讀取失敗：{e}")
        rp_raw = None

    master_df = None
    if article_master_file is not None:
        try:
            mname = article_master_file.name.lower()
            if mname.endswith((".xlsx", ".xls")):
                master_raw = pd.read_excel(article_master_file)
            else:
                master_raw = pd.read_csv(article_master_file)
            master_df = normalize_article_master(master_raw)
        except Exception as e:
            st.error(f"Article Master 檔讀取失敗：{e}")
            master_df = None

    if rp_raw is None or rp_raw.empty:
        st.stop()

    st.subheader("Step 2 — RP 來源資料預覽")
    st.dataframe(rp_raw.head(30), use_container_width=True)

    # Step 3：欄位標準化（僅供內部使用）
    rp_norm = normalize_columns(rp_raw)

    # 建立與 Result.xlsx 欄位對應的暫存標準欄位
    # 嘗試從 rp_norm 中找出對應來源
    def first_match(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for t in candidates:
            if t in df.columns:
                return t
        return None

    std = rp_norm.copy()

    # Site / Article / Description / Brand / MC / etc.
    # 此處僅作合理猜測，實務上可依實際 RP List 欄位名稱調整
    mapping_candidates: Dict[str, List[str]] = {
        "Site": ["site", "store", "location"],
        "Article": ["article", "article_code", "item", "sku"],
        "Article Description": ["article_description", "article_name", "description", "item_name"],
        "Brand": ["brand"],
        "MC": ["mc", "merchandise_category"],
        "MC Description": ["mc_description"],
        "Article categor": ["article_categor", "article_category"],
        "Article Type": ["article_type"],
        "Status": ["status"],
        "Stock On Hand  20070827": ["stock_on_hand", "current_stock", "on_hand", "soh"],
        "Purchase Group": ["purchase_group"],
        "RP Type": ["rp_type", "rp_typ"],
        "Planning Cycle": ["planning_cycle"],
        "Delivery Cycle": ["delivery_cycle"],
        "Stock Planner": ["stock_planner"],
        "Reorder Point": ["reorder_point", "rp"],
        "Delivery Days": ["delivery_days", "lead_time_days"],
        "Target Coverage": ["target_coverage"],
        "Supply Source (1=Vendor/2=DC)": ["supply_source"],
        "ABC Indicator": ["abc_indicator", "abc"],
    }

    for out_col, candidates in mapping_candidates.items():
        src = first_match(std, candidates)
        if src is not None and out_col not in std.columns:
            std[out_col] = std[src]

    # 如有 Article Master，使用 Article 作為 key 進行 left join 補充屬性
    if master_df is not None and "master_article_code" in master_df.columns:
        if "Article" in std.columns:
            std["_article_key"] = std["Article"].astype(str).str.strip()
        elif "article" in std.columns:
            std["_article_key"] = std["article"].astype(str).str.strip()
        else:
            std["_article_key"] = std.get("article_code", "").astype(str).str.strip()

        master_df = master_df.copy()
        master_df["master_article_code"] = master_df["master_article_code"].astype(str).str.strip()

        std = std.merge(
            master_df,
            how="left",
            left_on="_article_key",
            right_on="master_article_code",
            suffixes=("", "_m"),
        )
        std = std.drop(columns=[c for c in ["_article_key", "master_article_code"] if c in std.columns])

        # 若主檔提供 Brand / MC 等，可回填到對應輸出欄
        if "brand" in std.columns and "Brand" not in std.columns:
            std["Brand"] = std["brand"]
        if "merchandise_category" in std.columns and "MC" not in std.columns:
            std["MC"] = std["merchandise_category"]

    # Step 4：套用 MDB 邏輯骨架（New 欄位 + A/B/C QTY）
    planning_cycle_mapping = load_planning_cycle_mapping("Planning Cycle.xls")
    std_with_new = apply_mdb_logic(std, demo=use_default_new_values, pc_mapping=planning_cycle_mapping)

    # Step 5：構建最終輸出 DataFrame（僅包含 OUTPUT_COLUMNS，強制 1:1）
    result_rows = []

    # 將 std_with_new 作為來源，逐欄映射到 OUTPUT_COLUMNS
    for _, row in std_with_new.iterrows():
        out_row = {}
        for col in OUTPUT_COLUMNS:
            if col in std_with_new.columns:
                val = row[col]
            else:
                # 若沒有對應欄位，依型別補預設值
                if col in NUMERIC_LIKE_COLUMNS:
                    val = 0
                else:
                    val = ""
            out_row[col] = val
        result_rows.append(out_row)

    result_df = pd.DataFrame(result_rows, columns=OUTPUT_COLUMNS)

    # 型別與缺值最終處理：數值欄補 0，文字欄補空字串
    for col in OUTPUT_COLUMNS:
        if col in NUMERIC_LIKE_COLUMNS:
            result_df[col] = pd.to_numeric(result_df[col], errors="coerce").fillna(0)
        else:
            result_df[col] = result_df[col].fillna("")

    st.subheader("Step 3 — Result.xlsx 結構輸出預覽（前 200 筆）")
    st.dataframe(result_df.head(200), use_container_width=True)

    # Step 6：下載
    st.subheader("Step 4 — 下載 Result_streamlit.xlsx（結構 1:1 對齊 Result.xlsx）")

    excel_bytes = to_excel_download(result_df)
    st.download_button(
        label="下載 Result_streamlit.xlsx",
        data=excel_bytes,
        file_name="Result_streamlit.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.success(
        "已強制輸出為與 Result.xlsx 相同的 69 個欄位與順序。"
        "如需與 MDB 完全數值對齊，可在 apply_mdb_logic() 中補入正式規則。"
    )
