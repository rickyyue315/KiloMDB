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
# 欄位與計算設定（可依 MDB 調整）
# =============================

# 基本必備欄位（用於識別與計算）
REQUIRED_COLUMNS = [
    "article_code",   # 貨品編號（舊貨品檔中的貨號，將用來與 Article Master 對應）
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

# Article Master 欄位標準化目標：
# 以 SAP Article Number / SKUs Number 為鍵，補充屬性維度供篩選與輸出。
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


# =============================
# 欄位標題標準化：容忍不同命名
# =============================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    將「交易/庫存/銷量檔」欄位轉為內部統一命名：
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


def normalize_article_master(df: pd.DataFrame) -> pd.DataFrame:
    """
    專門處理 Article Master：
    - 標準化欄位名稱
    - 提取關鍵鍵值與屬性欄位
    - 回傳可供 join 使用的 master_df
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
        # key 候選
        "article_number_sap": "article_number_sap",
        "article_number__sap_": "article_number_sap",
        "article_no_sap": "article_number_sap",
        "skus_number_magic_sys": "skus_number_magic_sys",
        "skus_no_magic_sys": "skus_number_magic_sys",
        # 常用屬性
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

    # 嘗試建立 master_article_code 作為 join key（優先 SAP，其次 SKUs）
    key_col = None
    for cand in MASTER_KEY_CANDIDATES:
        if cand in m.columns:
            key_col = cand
            break

    if key_col is not None:
        m["master_article_code"] = m[key_col].astype(str).str.strip()
    else:
        # 若無可用 key，則回傳原樣（前端會檢查）
        m.attrs["original_columns"] = original_cols
        return m

    # 只保留 key + 屬性欄位，避免重覆欄位污染
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
# 讀取 Planning Cycle 對照表
# =============================
def load_planning_cycle_mapping(path_or_buffer) -> pd.DataFrame:
    """
    讀取 Planning Cycle.xls，並標準化為可用 mapping：

    預期欄位（不分大小寫，會做 alias）示例：
    - rp_type
    - old_planning_cycle
    - old_delivery_cycle
    - new_planning_cycle
    - new_delivery_cycle

    實作策略：
    - 僅在檔案成功讀取且核心欄位存在時啟用 mapping。
    - 其餘情況回傳空 df，呼叫端自行略過。
    """
    try:
        # 支援本地路徑與上傳檔 (BytesIO)
        pc_raw = pd.read_excel(path_or_buffer)
    except Exception:
        return pd.DataFrame()

    df = pc_raw.copy()
    df.columns = (
        df.columns.astype(str)
        .strip()
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

    # 必須欄位檢查：rp_type + old_planning_cycle + new_planning_cycle
    if not (col_rp and col_old_pc and col_new_pc):
        return pd.DataFrame()

    # 建立標準欄位
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
        # 若對照表未提供 New Delivery Cycle，預設沿用原 Delivery Cycle
        m["new_delivery_cycle"] = ""

    # 去重，避免多餘映射
    m = m.drop_duplicates(
        subset=["rp_type", "old_planning_cycle", "old_delivery_cycle"],
        keep="last",
    )

    return m


# =============================
# Streamlit 畫面與流程設計
# =============================

st.title("Old Article RP Parameter (Ideal Stock Added) — Streamlit版")

st.markdown(
    """
此程式目標：**重現 MDB 工具輸出 Result.xlsx 的欄位與邏輯**，可直接部署於 Streamlit Cloud。

原流程特徵（由 Result.xlsx 推斷）：
- 以 Site + Article 為粒度列出所有組合。
- 含完整主檔與規劃欄位（Brand、MC、Article Type、Status、ABC、RP Type、Planning Cycle、Delivery Cycle 等）。
- 同一組 Site + Article 同時顯示「舊設定」與「New ...」欄位（New Safety Qty / New Reorder Point / New Supply Source 等）。
- A/B/C QTY 欄位：對應不同方案 / 類型的建議量。

本版 Streamlit 程式設計原則：
- **輸入**：RP List / Article Master / 其他來源（格式需與現有流程一致或可映射）。
- **處理**：欄位標準化、依 MDB 邏輯計算新參數與建議量。
- **輸出**：欄位順序、欄位名稱、列順序盡量與 Result.xlsx 對齊，便於 1:1 比對。
- 若部分 MDB 內嵌邏輯無法完全從 Result.xlsx 反推，程式中以清楚註解標示「可依內部文件補完公式」。

下方為簡化版 1:1 對齊實作骨架：
- 僅使用明顯可從 Result.xlsx 推論的欄位與關係。
- 預留 `apply_mdb_logic()` 區塊集中管理邏輯，方便後續依你提供的 MDB / 規則逐條補齊。
"""
)

# -------- Sidebar：上傳來源檔與控制參數 --------
with st.sidebar:
    st.header("Step 1 — 上傳來源資料")

    rp_source_file = st.file_uploader(
        "上傳 RP List / 原始來源 (建議：與產出此 Result.xlsx 同一來源)",
        type=["xlsx", "xls", "csv", "txt"],
        help="用來重建 Site + Article 粒度資料列及原始 RP 等欄位。",
    )

    article_master_file = st.file_uploader(
        "（可選）上傳 Article Master 檔",
        type=["xlsx", "xls", "csv"],
        help="若 RP 來源未含完整主檔資訊，可用此補足。",
    )

    st.markdown("---")
    st.subheader("運算控制（暫以預設，細節在 apply_mdb_logic 調整）")

    use_default_new_values = st.checkbox(
        "啟用示範新參數計算（未提供 MDB 完整公式時）", value=True
    )

    st.caption(
        "最終要做到與 MDB 1:1 對齊時，只需在程式內補上正式公式與欄位對應，"
        "不需要改動使用者操作流程。"
    )

# -------- 若未上傳 RP 來源，提示並結束 --------
if not rp_source_file:
    st.info("請先在左側上傳 RP List / 原始來源檔，再進行比對與輸出。")
else:
    # -------- Step 2：讀取 RP 來源 --------
    try:
        name = rp_source_file.name.lower()
        if name.endswith((".xlsx", ".xls")):
            rp_raw = pd.read_excel(rp_source_file)
        elif name.endswith(".txt"):
            # 多數 RP List.txt 為固定欄寬或多空白分隔，這裡先用 delim_whitespace 示範
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

    # -------- Step 3：欄位標準化（RP 來源） --------
    rp_norm = normalize_columns(rp_raw)

    # 強制建立核心鍵：site, article, article_description 等（依 Result.xlsx 欄位）
    # 這裡假設 RP 檔至少有 Site 與 Article 欄位或可映射，若實務不同，可在此擴充 mapping。
    col_map = {
        "site": ["site", "store", "location"],
        "article": ["article", "article_code", "item", "sku"],
        "article_description": ["article_description", "article_name", "description"],
        "brand": ["brand"],
        "mc": ["mc", "merchandise_category"],
        "mc_description": ["mc_description"],
        "article_categor": ["article_categor", "article_category"],
        "article_type": ["article_type"],
        "status": ["status"],
        "stock_on_hand": ["stock_on_hand", "current_stock", "on_hand", "soh"],
        "reorder_point": ["reorder_point", "rp", "old_reorder_point"],
        "delivery_days": ["delivery_days", "lead_time_days"],
        "target_coverage": ["target_coverage"],
        "supply_source": ["supply_source"],
        "abc_indicator": ["abc_indicator", "abc"],
    }

    def first_match(df, targets):
        for t in targets:
            if t in df.columns:
                return t
        return None

    # 建立標準欄位（若有對應）
    std = rp_norm.copy()
    for std_col, candidates in col_map.items():
        src = first_match(std, candidates)
        if src is not None and std_col not in std.columns:
            std[std_col] = std[src]

    # 確保 Site, Article 存在
    if "site" not in std.columns or "article" not in std.columns:
        st.error("RP 來源中缺少必備鍵欄位 Site 或 Article，請確認來源格式或在程式中補上 mapping。")
        st.stop()

    # -------- Step 4：如有 Article Master，依 Article 鍵做 left join 補齊屬性 --------
    if master_df is not None and "master_article_code" in master_df.columns:
        std["article_key"] = std["article"].astype(str).str.strip()
        master_df["master_article_code"] = master_df["master_article_code"].astype(str).str.strip()
        std = std.merge(
            master_df,
            how="left",
            left_on="article_key",
            right_on="master_article_code",
            suffixes=("", "_m"),
        )
        std = std.drop(columns=["article_key", "master_article_code"])

    # -------- Step 5：套用 MDB 對應邏輯：產生 New 欄位與 A/B/C QTY --------
    # 預先嘗試載入 Planning Cycle 對照表（若專案根目錄存在則啟用）
    planning_cycle_mapping = load_planning_cycle_mapping("Planning Cycle.xls")

    def apply_mdb_logic(
        df: pd.DataFrame,
        demo: bool = True,
        pc_mapping: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        專門對應 MDB -> Result.xlsx 的欄位與計算。

        New Planning Cycle / New Delivery Cycle：
        - 依據使用者提供的 Planning Cycle.xls。
        - 預期以 (RP Type, Planning Cycle, Delivery Cycle) 為 key 做 mapping。
        - 若無 Delivery Cycle，則以 (RP Type, Planning Cycle) 為 key。

        其他欄位：
        - 目前仍採安全示範公式，結構對齊，之後可按 MDB 真實規則覆寫。
        """
        df = df.copy()

        # 轉數值欄位
        for c in [
            "stock_on_hand",
            "reorder_point",
            "delivery_days",
            "target_coverage",
        ]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # 正規化文字鍵
        def norm_str(s: pd.Series) -> pd.Series:
            return s.astype(str).str.strip()

        # -----------------------------
        # 1) New Planning/Delivery Cycle by mapping
        # -----------------------------
        if pc_mapping is not None and not pc_mapping.empty:
            # 準備來源鍵欄位（RP Type / Planning Cycle / Delivery Cycle）
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

                # 完整 key join: (rp_type, planning_cycle, delivery_cycle)
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

                # 若 delivery_cycle 維度無法匹配，退而只用 (rp_type, planning_cycle)
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
                    # 優先使用三鍵匹配，其次兩鍵匹配
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

                # 若對照表未提供 New Delivery Cycle，則沿用原 Delivery Cycle
                if "New Delivery Cycle" in df.columns:
                    df["New Delivery Cycle"] = df["New Delivery Cycle"].where(
                        df["New Delivery Cycle"].notna(),
                        df.get("Delivery Cycle", df.get("delivery_cycle", "")),
                    )

                # 清理暫存鍵與中間欄位
                drop_cols = [
                    c
                    for c in df.columns
                    if c.startswith("_rp_type_key")
                    or c.startswith("_pc_key")
                    or c.startswith("_dc_key")
                    or c in ["new_planning_cycle", "new_delivery_cycle", "new_planning_cycle_pcmap2", "new_delivery_cycle_pcmap2"]
                ]
                df = df.drop(columns=[c for c in drop_cols if c in df.columns])
            else:
                # 若缺 RP Type / Planning Cycle，則不套用 mapping，交給預設邏輯
                pass

        # -----------------------------
        # 2) New Safety Qty 示範邏輯
        # -----------------------------
        if demo:
            if "reorder_point" in df.columns and "stock_on_hand" in df.columns:
                df["New Safety Qty"] = (
                    df["reorder_point"] - df["stock_on_hand"]
                ).clip(lower=0).fillna(0)
            else:
                df["New Safety Qty"] = df.get("New Safety Qty", 0)
        else:
            df["New Safety Qty"] = df.get("New Safety Qty", 0)

        # -----------------------------
        # 3) 其餘 New* 欄位預設沿用舊欄位，維持結構對齊
        # -----------------------------
        mapping_pairs = [
            ("Purchase Group", "New Purchase Group"),
            ("RP Type", "New RP Typ"),
            ("Stock Planner", "New Stock Planner"),
            ("Reorder Point", "New Reorder Point"),
            ("Delivery Days", "New Delivery Days"),
            ("Target Coverage", "New Traget Coverage"),
            ("Supply Source (1=Vendor/2=DC)", "New Supply Source"),
            ("ABC Indicator", "New ABC Indicator"),
        ]
        for src, tgt in mapping_pairs:
            if tgt not in df.columns:
                if src in df.columns:
                    df[tgt] = df[src]
                else:
                    df[tgt] = ""

        # 若尚未由 mapping 計算 New Planning Cycle / New Delivery Cycle，則沿用原值
        if "New Planning Cycle" not in df.columns:
            base_pc = "Planning Cycle" if "Planning Cycle" in df.columns else "planning_cycle"
            df["New Planning Cycle"] = df.get(base_pc, "")
        if "New Delivery Cycle" not in df.columns:
            base_dc = "Delivery Cycle" if "Delivery Cycle" in df.columns else "delivery_cycle"
            df["New Delivery Cycle"] = df.get(base_dc, "")

        # -----------------------------
        # 4) New* 相關預設欄位填補
        # -----------------------------
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

        # -----------------------------
        # 5) A/B/C QTY 示範邏輯
        # -----------------------------
        if demo:
            if "New Reorder Point" in df.columns and "stock_on_hand" in df.columns:
                base = (
                    df["New Reorder Point"] - df["stock_on_hand"]
                ).clip(lower=0).fillna(0)
            else:
                base = 0
            df["A QTY"] = df.get("A QTY", base)
            df["B QTY"] = df.get("B QTY", base)
            df["C QTY"] = df.get("C QTY", base)
        else:
            for col in ["A QTY", "B QTY", "C QTY"]:
                if col not in df.columns:
                    df[col] = 0

        return df

    result_df = apply_mdb_logic(std, demo=use_default_new_values, pc_mapping=planning_cycle_mapping)

    # -------- Step 6：欄位順序盡量對齊 Result.xlsx --------
    desired_order = [
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
        "New Traget Coverage",
        "New Supply Source",
        "New ABC Indicator",
        "New Smoothing (0/1)",
        "New Forecast Model",
        "New Historical perio",
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

    # 嘗試用不區分大小寫的方式重排欄位
    col_lower_map = {c.lower(): c for c in result_df.columns}
    ordered_cols = []
    for col in desired_order:
        key = col.lower()
        if key in col_lower_map:
            ordered_cols.append(col_lower_map[key])
    # 加入未在 desired_order 中，但出現在 result_df 的欄位
    ordered_cols += [c for c in result_df.columns if c not in ordered_cols]
    result_df = result_df[ordered_cols]

    st.subheader("Step 3 — 模擬 Result.xlsx 結構的輸出結果")
    st.dataframe(result_df.head(200), use_container_width=True)

    # -------- Step 7：下載成 Result.xlsx 格式 --------
    st.subheader("Step 4 — 匯出 Result.xlsx 格式")

    excel_bytes = to_excel_download(result_df)
    st.download_button(
        label="下載模擬 Result.xlsx（用於與 MDB 原輸出比對）",
        data=excel_bytes,
        file_name="Result_streamlit.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.success(
        "已依目前可見資訊重建 Result.xlsx 欄位結構。若你提供 MDB 內部公式/規則，可直接在 apply_mdb_logic() 補上以達成 1:1 數值對齊。"
    )
