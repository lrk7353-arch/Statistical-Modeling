# -*- coding: utf-8 -*-
"""
Shanghai public-data catalog enrichment and dormant-asset pipeline.

This script is intentionally split into stages:

1. build-master: read the official catalog Excel and create a stable master table.
2. collect-list-formats: connect to a user-opened Chrome session and collect visible
   format labels from list pages.
3. collect-detail-fields: connect to the same Chrome session and collect public
   detail-page fields with checkpoint/resume support.
4. fit-model: build an sklearn potential-use model and compute dormant scores.
5. export-workbook: assemble a human-readable result workbook.

The browser stages do not bypass anti-crawling controls. They use a visible browser
that the user starts and logs into, run single-threaded, and save progress often.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import gettempdir
from typing import Iterable
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm


BASE_URL = "https://data.sh.gov.cn"
LIST_URL = f"{BASE_URL}/view/data-resource/index.html"
DEFAULT_CATALOG = "上海市公共数据开放平台资源目录.xlsx"
DEFAULT_OUTPUT_DIR = "outputs"

REQUIRED_COLUMNS = [
    "序号",
    "数据资源提供部门",
    "数据集ID",
    "数据资源名称",
    "数据领域",
    "开放条件",
    "浏览量",
    "下载量",
    "接口调用量",
    "数据资源内容描述",
    "数据资源类型",
    "数据资源状态",
    "更新频率",
    "开放属性",
    "首次发布日期",
    "最近更新日期",
]

FORMAT_LABELS = ["RDF", "XML", "CSV", "JSON", "XLSX"]
FORMAT_COLUMNS = {
    "RDF": "has_rdf",
    "XML": "has_xml",
    "CSV": "has_csv",
    "JSON": "has_json",
    "XLSX": "has_xlsx",
}

SUPPLEMENT_COLUMNS = [
    "download_formats",
    "has_rdf",
    "has_xml",
    "has_csv",
    "has_json",
    "has_xlsx",
    "api_need_apply",
    "format_count",
    "record_count",
    "data_size",
    "field_names",
    "field_count",
    "has_time_field",
    "has_geo_field",
    "field_description_count",
    "has_standard_field_description",
    "has_data_sample",
    "sample_field_headers",
    "recommended_dataset_count",
    "recommended_dataset_names",
    "rating_score",
    "comment_count",
    "detail_spatial_scope",
    "detail_time_scope",
    "scrape_status",
    "scrape_error",
    "scraped_at",
    "备注",
]

DISTRICT_NAMES = [
    "浦东新区",
    "黄浦区",
    "徐汇区",
    "长宁区",
    "静安区",
    "普陀区",
    "虹口区",
    "杨浦区",
    "闵行区",
    "宝山区",
    "嘉定区",
    "金山区",
    "松江区",
    "青浦区",
    "奉贤区",
    "崇明区",
]

SHANGHAI_DISTRICTS = [
    "浦东新区",
    "黄浦区",
    "徐汇区",
    "长宁区",
    "静安区",
    "普陀区",
    "虹口区",
    "杨浦区",
    "闵行区",
    "宝山区",
    "嘉定区",
    "金山区",
    "松江区",
    "青浦区",
    "奉贤区",
    "崇明区",
]

TRADITIONAL_TO_SIMPLIFIED = str.maketrans(
    {
        "個": "个",
        "備": "备",
        "價": "价",
        "儀": "仪",
        "億": "亿",
        "內": "内",
        "兩": "两",
        "冊": "册",
        "寫": "写",
        "凍": "冻",
        "劃": "划",
        "劑": "剂",
        "辦": "办",
        "動": "动",
        "務": "务",
        "區": "区",
        "協": "协",
        "單": "单",
        "單": "单",
        "衛": "卫",
        "廠": "厂",
        "縣": "县",
        "參": "参",
        "號": "号",
        "員": "员",
        "啟": "启",
        "團": "团",
        "圖": "图",
        "國": "国",
        "園": "园",
        "圍": "围",
        "場": "场",
        "塊": "块",
        "報": "报",
        "壓": "压",
        "壞": "坏",
        "壯": "壮",
        "處": "处",
        "備": "备",
        "夾": "夹",
        "學": "学",
        "實": "实",
        "審": "审",
        "對": "对",
        "專": "专",
        "導": "导",
        "屬": "属",
        "崗": "岗",
        "幣": "币",
        "幹": "干",
        "庫": "库",
        "廳": "厅",
        "廢": "废",
        "廣": "广",
        "建": "建",
        "異": "异",
        "彙": "汇",
        "徵": "征",
        "態": "态",
        "應": "应",
        "戶": "户",
        "執": "执",
        "擇": "择",
        "據": "据",
        "擬": "拟",
        "擴": "扩",
        "數": "数",
        "斷": "断",
        "時": "时",
        "暫": "暂",
        "會": "会",
        "條": "条",
        "標": "标",
        "樣": "样",
        "檔": "档",
        "檢": "检",
        "欄": "栏",
        "權": "权",
        "歸": "归",
        "殘": "残",
        "氣": "气",
        "決": "决",
        "況": "况",
        "測": "测",
        "濟": "济",
        "為": "为",
        "無": "无",
        "營": "营",
        "狀": "状",
        "獎": "奖",
        "環": "环",
        "現": "现",
        "產": "产",
        "畫": "画",
        "登": "登",
        "發": "发",
        "監": "监",
        "碼": "码",
        "確": "确",
        "礎": "础",
        "禮": "礼",
        "種": "种",
        "稱": "称",
        "稅": "税",
        "穩": "稳",
        "窯": "窑",
        "筆": "笔",
        "範": "范",
        "類": "类",
        "糧": "粮",
        "約": "约",
        "級": "级",
        "組": "组",
        "經": "经",
        "綜": "综",
        "網": "网",
        "縣": "县",
        "總": "总",
        "織": "织",
        "統": "统",
        "維": "维",
        "聯": "联",
        "職": "职",
        "臺": "台",
        "與": "与",
        "舊": "旧",
        "萬": "万",
        "處": "处",
        "號": "号",
        "衆": "众",
        "補": "补",
        "裝": "装",
        "規": "规",
        "視": "视",
        "計": "计",
        "訊": "讯",
        "註": "注",
        "設": "设",
        "許": "许",
        "評": "评",
        "詢": "询",
        "該": "该",
        "詳": "详",
        "認": "认",
        "說": "说",
        "調": "调",
        "請": "请",
        "諮": "咨",
        "證": "证",
        "識": "识",
        "變": "变",
        "讓": "让",
        "財": "财",
        "責": "责",
        "貿": "贸",
        "費": "费",
        "資": "资",
        "賽": "赛",
        "贈": "赠",
        "趨": "趋",
        "車": "车",
        "軌": "轨",
        "轉": "转",
        "辦": "办",
        "農": "农",
        "運": "运",
        "過": "过",
        "還": "还",
        "適": "适",
        "選": "选",
        "鄉": "乡",
        "醫": "医",
        "釋": "释",
        "針": "针",
        "銷": "销",
        "鎮": "镇",
        "長": "长",
        "門": "门",
        "開": "开",
        "間": "间",
        "隊": "队",
        "階": "阶",
        "際": "际",
        "隨": "随",
        "難": "难",
        "電": "电",
        "項": "项",
        "領": "领",
        "類": "类",
        "風": "风",
        "體": "体",
        "點": "点",
    }
)

SCENE_KEYWORDS = [
    "地址",
    "时间",
    "日期",
    "企业",
    "机构",
    "设施",
    "项目",
    "金额",
    "人员",
    "人口",
    "道路",
    "交通",
    "医疗",
    "教育",
    "养老",
    "环境",
    "安全",
    "执法",
    "监管",
    "街道",
    "经纬度",
    "坐标",
    "区",
    "镇",
]

FIELD_NAME_BAD_VALUES = {
    "",
    "1",
    "token",
    "offset",
    "limit",
    "参数",
    "参数名",
    "参数名称",
    "参数描述",
    "参数类型",
    "字段",
    "字段名",
    "字段名称",
    "字段大小",
    "名称",
    "说明",
    "描述",
    "数据标签",
    "关键字",
    "关键词",
    "下载格式",
    "下載格式",
    "全选",
    "全選",
    "其他",
    "备注",
    "備註",
    "字元型C",
    "字符型C",
    "字符型",
    "数字型",
    "数值型",
    "日期型",
    "文本型",
    "时间戳",
    "字段描述",
    "字段類型",
    "字段类型",
    "序号",
    "序號",
    "数据大小",
    "數據大小",
    "數據標籤",
    "档案名称",
    "檔案名稱",
    "文件名称",
    "栏位名称",
    "欄位名稱",
    "栏位长度",
    "欄位長度",
    "關鍵字",
}

FIELD_NAME_HINTS = (
    "名称",
    "编号",
    "代码",
    "编码",
    "地址",
    "时间",
    "日期",
    "类型",
    "类别",
    "状态",
    "数量",
    "金额",
    "面积",
    "经度",
    "纬度",
    "坐标",
    "年份",
    "年度",
    "月份",
    "内容",
    "单位",
    "电话",
    "姓名",
    "区县",
    "街道",
    "机构",
    "企业",
    "项目",
    "证号",
    "级别",
    "来源",
    "结果",
    "描述",
    "说明",
    "备注",
    "标识",
    "地区",
    "区划",
    "科目",
    "网址",
    "邮箱",
    "序号",
    "ID",
    "id",
)


@dataclass
class PipelinePaths:
    base_dir: Path
    input_catalog: Path
    output_dir: Path

    @property
    def master_xlsx(self) -> Path:
        return self.output_dir / "dataset_master_enriched.xlsx"

    @property
    def checkpoint_csv(self) -> Path:
        return self.output_dir / "web_supplement_checkpoint.csv"

    @property
    def failed_csv(self) -> Path:
        return self.output_dir / "web_supplement_failed.csv"

    @property
    def list_unmatched_csv(self) -> Path:
        return self.output_dir / "list_format_unmatched.csv"

    @property
    def model_results_csv(self) -> Path:
        return self.output_dir / "model_results.csv"

    @property
    def model_metrics_json(self) -> Path:
        return self.output_dir / "model_metrics.json"

    @property
    def dormant_workbook(self) -> Path:
        return self.output_dir / "dormant_asset_results.xlsx"

    @property
    def minimum_required_xlsx(self) -> Path:
        return self.output_dir / "minimum_required_fields.xlsx"


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_path(value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir or script_dir()) / path


def make_paths(args: argparse.Namespace) -> PipelinePaths:
    base_dir = script_dir()
    input_catalog = resolve_path(getattr(args, "input", DEFAULT_CATALOG), base_dir)
    output_dir = resolve_path(getattr(args, "output_dir", DEFAULT_OUTPUT_DIR), base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return PipelinePaths(base_dir=base_dir, input_catalog=input_catalog, output_dir=output_dir)


def normalize_space(text: object) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def console_safe(text: object) -> str:
    value = str(text)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return value.encode(encoding, errors="replace").decode(encoding, errors="replace")


_OPENCC_CONVERTER = None


def to_simplified_text(text: object) -> str:
    value = normalize_space(text)
    if not value:
        return ""
    global _OPENCC_CONVERTER
    if _OPENCC_CONVERTER is None:
        try:
            from opencc import OpenCC

            _OPENCC_CONVERTER = OpenCC("t2s")
        except Exception:
            _OPENCC_CONVERTER = False
    if _OPENCC_CONVERTER:
        try:
            return normalize_space(_OPENCC_CONVERTER.convert(value))
        except Exception:
            pass
    return normalize_space(value.translate(TRADITIONAL_TO_SIMPLIFIED).replace("瀏覽", "浏览"))


def normalize_id(value: object) -> str:
    text = normalize_space(value)
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def read_catalog(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")
    df = pd.read_excel(path, dtype={"数据集ID": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Catalog is missing required columns: {missing}")
    df["数据集ID"] = df["数据集ID"].map(normalize_id)
    return df


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def percentile_rank(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    return numeric.rank(method="average", pct=True)


def safe_int(value: object) -> int | None:
    text = normalize_space(value)
    if not text or text in {"-", "--", "无", "暂无", "null", "None"}:
        return None
    match = re.search(r"[\d,]+", text)
    if not match:
        return None
    return int(match.group(0).replace(",", ""))


def resource_type_code(resource_type: object) -> str:
    text = normalize_space(resource_type)
    if text == "数据产品":
        return "cp"
    if text == "数据接口":
        return "jk"
    return ""


def build_detail_url(dataset_id: object, type_code: str) -> str:
    did = normalize_id(dataset_id)
    if not did or not type_code:
        return ""
    return f"{BASE_URL}/view/detail/index.html?type={type_code}&&id={did}&&companyFlag=0"


def infer_spatial_admin_level(department: object) -> str:
    text = normalize_space(department)
    if not text:
        return "未知"
    if any(word in text for word in ["街道", "镇人民政府", "乡人民政府"]):
        return "街镇级"
    if any(name in text for name in DISTRICT_NAMES):
        return "区级"
    if text.startswith("上海市") or "上海市" in text:
        return "市级"
    if any(word in text for word in ["长三角", "国家", "中国"]):
        return "跨区域/国家级"
    return "其他/不确定"


def infer_spatial_scope_from_catalog(name: object, department: object = "", summary: object = "") -> str:
    text = to_simplified_text(" ".join([normalize_space(name), normalize_space(department), normalize_space(summary)]))
    if not text:
        return ""
    found = [district for district in SHANGHAI_DISTRICTS if district in text]
    if found:
        return ";".join(dict.fromkeys(found))
    if "上海市" in text or text.startswith("市"):
        return "上海市"
    return ""


def add_catalog_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["type_code"] = out["数据资源类型"].map(resource_type_code)
    out["detail_url"] = [
        build_detail_url(dataset_id, type_code)
        for dataset_id, type_code in zip(out["数据集ID"], out["type_code"])
    ]
    out["spatial_admin_level"] = out["数据资源提供部门"].map(infer_spatial_admin_level)
    out["inferred_spatial_scope"] = [
        infer_spatial_scope_from_catalog(name, dept, summary)
        for name, dept, summary in zip(out["数据资源名称"], out["数据资源提供部门"], out["数据资源内容描述"])
    ]

    first_dt = pd.to_datetime(out["首次发布日期"], errors="coerce")
    update_dt = pd.to_datetime(out["最近更新日期"], errors="coerce")
    today = pd.Timestamp.today().normalize()
    out["first_publish_date_parsed"] = first_dt
    out["last_update_date_parsed"] = update_dt
    out["publication_age_days"] = (today - first_dt).dt.days
    out["update_recency_days"] = (today - update_dt).dt.days
    out["maintenance_span_days"] = (update_dt - first_dt).dt.days

    out["title_len"] = out["数据资源名称"].fillna("").astype(str).str.len()
    out["description_len"] = out["数据资源内容描述"].fillna("").astype(str).str.len()
    out["description_has_field_hint"] = out["数据资源内容描述"].fillna("").astype(str).str.contains(
        "字段|主要包含|包含了|包括", regex=True
    ).astype(int)
    out["scene_keyword_count"] = out["数据资源内容描述"].fillna("").astype(str).map(
        lambda x: sum(1 for kw in SCENE_KEYWORDS if kw in x)
    )

    for col in ["浏览量", "下载量", "接口调用量"]:
        out[col] = safe_numeric(out[col]).fillna(0)
        out[f"log_{col}"] = np.log1p(out[col])
        out[f"{col}_percentile"] = percentile_rank(out[f"log_{col}"])

    out["ActualUse_v0"] = (
        0.45 * out["浏览量_percentile"]
        + 0.35 * out["下载量_percentile"]
        + 0.20 * out["接口调用量_percentile"]
    )
    out["ActualUse_percentile"] = percentile_rank(out["ActualUse_v0"])

    for col in SUPPLEMENT_COLUMNS:
        if col not in out.columns:
            if col.startswith("has_") or col.endswith("_count") or col in {"rating_score", "record_count", "api_need_apply"}:
                out[col] = np.nan
            else:
                out[col] = ""

    return out


def write_master(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="dataset_master")


def build_master(args: argparse.Namespace) -> None:
    paths = make_paths(args)
    df = read_catalog(paths.input_catalog)
    enriched = add_catalog_features(df)
    write_master(enriched, paths.master_xlsx)
    export_minimum_required(paths, enriched)
    print(f"[ok] rows={len(enriched)}")
    print(f"[ok] type_code counts={enriched['type_code'].value_counts(dropna=False).to_dict()}")
    print(f"[ok] saved {paths.master_xlsx}")
    print(f"[ok] saved {paths.minimum_required_xlsx}")


def read_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Master workbook not found: {path}. Run build-master first.")
    df = pd.read_excel(path, dtype={"数据集ID": str})
    df["数据集ID"] = df["数据集ID"].map(normalize_id)
    return df


def read_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["数据集ID"])
    df = pd.read_csv(path, dtype={"数据集ID": str}, encoding="utf-8-sig")
    if "数据集ID" in df.columns:
        df["数据集ID"] = df["数据集ID"].map(normalize_id)
    return df


TERMINAL_SCRAPE_STATUSES = {"success", "unavailable"}
UNAVAILABLE_PAGE_MARKERS = [
    "请检查您输入的数据集id是否正确",
    "由于该数据集已被下架导致无法继续访问",
    "数据集已被下架",
]


def is_terminal_scrape_status(value: object) -> bool:
    return normalize_space(value).lower() in TERMINAL_SCRAPE_STATUSES


def is_unavailable_detail_page(text: str) -> bool:
    simple = to_simplified_text(text)
    return any(marker in simple for marker in UNAVAILABLE_PAGE_MARKERS)


def unavailable_detail_record(dataset_id: str, url: str, text: str) -> dict:
    return {
        "数据集ID": normalize_id(dataset_id),
        "detail_url": url,
        "download_formats": "",
        "api_need_apply": "",
        "format_count": 0,
        "record_count": "",
        "data_size": "",
        "field_names": "",
        "field_count": 0,
        "has_time_field": 0,
        "has_geo_field": 0,
        "field_description_count": 0,
        "has_standard_field_description": 0,
        "has_data_sample": 0,
        "sample_field_headers": "",
        "recommended_dataset_count": 0,
        "recommended_dataset_names": "",
        "rating_score": "",
        "comment_count": "",
        "detail_spatial_scope": "",
        "detail_time_scope": "",
        "scrape_status": "unavailable",
        "scrape_error": "detail page says dataset id is invalid or removed",
        "scraped_at": datetime.now().isoformat(timespec="seconds"),
        "备注": "详情页提示数据集ID不正确或数据集已下架，作为终止状态跳过重试",
    }


def upsert_checkpoint(path: Path, records: Iterable[dict]) -> pd.DataFrame:
    new_df = pd.DataFrame(list(records))
    if new_df.empty:
        return read_checkpoint(path)
    if "数据集ID" not in new_df.columns:
        raise ValueError("Checkpoint records must include 数据集ID.")
    new_df["数据集ID"] = new_df["数据集ID"].map(normalize_id)
    old_df = read_checkpoint(path)
    old_df = old_df.dropna(axis=1, how="all")
    new_df = new_df.dropna(axis=1, how="all")
    latest_columns = {"scrape_status", "scrape_error", "scraped_at", "备注"}

    if old_df.empty:
        combined = new_df.copy()
        combined.to_csv(path, index=False, encoding="utf-8-sig")
        return combined

    all_columns = list(old_df.columns)
    all_columns.extend(col for col in new_df.columns if col not in old_df.columns)
    combined = old_df.reindex(columns=all_columns).copy()
    new_df = new_df.reindex(columns=all_columns)
    index_by_id = {normalize_id(value): idx for idx, value in combined["数据集ID"].items()}
    appended_rows = []

    def has_non_empty_value(value: object) -> bool:
        if pd.isna(value):
            return False
        return normalize_space(value) != ""

    for _, row in new_df.iterrows():
        dataset_id = normalize_id(row["数据集ID"])
        if dataset_id in index_by_id:
            idx = index_by_id[dataset_id]
            for col, value in row.items():
                if col == "数据集ID" or col in latest_columns:
                    combined.at[idx, col] = value
                elif has_non_empty_value(value):
                    combined.at[idx, col] = value
        else:
            appended_rows.append(row)
            index_by_id[dataset_id] = len(combined) + len(appended_rows) - 1

    if appended_rows:
        combined = pd.concat([combined, pd.DataFrame(appended_rows)], ignore_index=True, sort=False)
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return combined


def merge_checkpoint_into_master(master: pd.DataFrame, checkpoint: pd.DataFrame) -> pd.DataFrame:
    if checkpoint.empty or "数据集ID" not in checkpoint.columns:
        return master
    cp = checkpoint.drop_duplicates(subset=["数据集ID"], keep="last").copy()
    merged = master.merge(cp, on="数据集ID", how="left", suffixes=("", "__cp"))
    for col in cp.columns:
        if col == "数据集ID":
            continue
        cp_col = f"{col}__cp" if col in master.columns else col
        if cp_col not in merged.columns:
            continue
        if col not in merged.columns:
            merged[col] = merged[cp_col]
        else:
            mask = merged[cp_col].notna() & (merged[cp_col].astype(str) != "")
            if mask.any():
                merged[col] = merged[col].astype("object")
            merged.loc[mask, col] = merged.loc[mask, cp_col]
        if cp_col != col:
            merged = merged.drop(columns=[cp_col])
    return merged


def sync_master_from_checkpoint(paths: PipelinePaths) -> pd.DataFrame:
    master = read_master(paths.master_xlsx)
    checkpoint = read_checkpoint(paths.checkpoint_csv)
    merged = merge_checkpoint_into_master(master, checkpoint)
    if "field_names" in merged.columns:
        merged["field_names"] = merged["field_names"].map(clean_field_names_text)
        merged["field_count"] = merged["field_names"].map(
            lambda value: 0 if not normalize_space(value) else len(normalize_space(value).split(";"))
        )
    write_master(merged, paths.master_xlsx)
    export_minimum_required(paths, merged)
    return merged


def value_or_blank(df: pd.DataFrame, col: str):
    if col in df.columns:
        return df[col]
    return ""


def first_nonempty_series(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series([""] * len(df), index=df.index, dtype="object")
    for col in columns:
        if col not in df.columns:
            continue
        values = df[col].fillna("").astype(str)
        mask = (result.astype(str) == "") & (values != "")
        result.loc[mask] = values.loc[mask]
    return result


def export_minimum_required(paths: PipelinePaths, df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = read_master(paths.master_xlsx)
    out = pd.DataFrame()
    out["数据集ID"] = value_or_blank(df, "数据集ID")
    out["数据资源名称"] = value_or_blank(df, "数据资源名称")
    out["dataset_url"] = value_or_blank(df, "detail_url")
    out["download_formats"] = value_or_blank(df, "download_formats")
    out["has_csv"] = value_or_blank(df, "has_csv")
    out["has_json"] = value_or_blank(df, "has_json")
    out["has_xlsx"] = value_or_blank(df, "has_xlsx")
    out["has_xml"] = value_or_blank(df, "has_xml")
    out["has_rdf"] = value_or_blank(df, "has_rdf")
    out["has_api_doc"] = (value_or_blank(df, "数据资源类型").astype(str) == "数据接口").astype(int) if "数据资源类型" in df.columns else ""
    out["api_need_apply"] = value_or_blank(df, "api_need_apply")
    out["record_count"] = value_or_blank(df, "record_count")
    out["data_size"] = value_or_blank(df, "data_size").map(to_simplified_text) if "data_size" in df.columns else ""
    out["spatial_scope"] = first_nonempty_series(df, ["detail_spatial_scope", "inferred_spatial_scope"]).map(to_simplified_text)
    out["time_scope"] = value_or_blank(df, "detail_time_scope").map(to_simplified_text) if "detail_time_scope" in df.columns else ""
    out["field_names"] = value_or_blank(df, "field_names").map(clean_field_names_text) if "field_names" in df.columns else ""
    out["field_count"] = out["field_names"].map(
        lambda value: 0 if not normalize_space(value) else len(normalize_space(value).split(";"))
    )
    out["has_time_field"] = value_or_blank(df, "has_time_field")
    out["has_geo_field"] = value_or_blank(df, "has_geo_field")
    out["has_data_sample"] = value_or_blank(df, "has_data_sample")
    out["recommended_dataset_count"] = value_or_blank(df, "recommended_dataset_count")
    out["备注"] = value_or_blank(df, "备注")
    with pd.ExcelWriter(paths.minimum_required_xlsx, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="minimum_required")
    return out


def print_chrome_help() -> None:
    print(
        "\nStart a visible Chrome with remote debugging before browser collection, for example:\n"
        '  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" '
        '--remote-debugging-port=9222 --user-data-dir="%TEMP%\\sh-data-browser"\n'
        "Then log in to data.sh.gov.cn in that Chrome window and rerun this command.\n"
    )


class CDPChrome:
    """Small Chrome DevTools Protocol client used when chromedriver is unavailable."""

    def __init__(self, debugger_address: str, navigate_wait_seconds: float = 20.0):
        try:
            import websocket
        except ImportError as exc:
            raise RuntimeError("websocket-client is required for CDP browser collection.") from exc

        self.debugger_address = debugger_address
        self._websocket = websocket
        self._next_id = 1
        self.ws = None
        self.navigate_wait_seconds = max(0.5, float(navigate_wait_seconds or 20.0))
        self._connect()
        self.send("Page.enable")
        self.send("Runtime.enable")

    def _http_json(self, path: str):
        url = f"http://{self.debugger_address}{path}"
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def _connect(self) -> None:
        targets = self._http_json("/json/list")
        page_targets = [t for t in targets if t.get("type") == "page" and t.get("webSocketDebuggerUrl")]
        if not page_targets:
            raise RuntimeError("No Chrome page target found. Open a normal tab first.")
        target = page_targets[0]
        self.ws = self._websocket.create_connection(
            target["webSocketDebuggerUrl"],
            timeout=10,
            suppress_origin=True,
        )

    def send(self, method: str, params: dict | None = None) -> dict:
        if self.ws is None:
            raise RuntimeError("CDP websocket is not connected.")
        message_id = self._next_id
        self._next_id += 1
        self.ws.send(json.dumps({"id": message_id, "method": method, "params": params or {}}))
        while True:
            raw = self.ws.recv()
            message = json.loads(raw)
            if message.get("id") == message_id:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message.get("result", {})

    def evaluate(self, expression: str):
        result = self.send(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": True,
            },
        )
        remote = result.get("result", {})
        return remote.get("value")

    def get(self, url: str) -> None:
        self.send("Page.navigate", {"url": url})
        deadline = time.time() + self.navigate_wait_seconds
        started_at = time.time()
        target_id = parse_dataset_id_from_url(url)
        while time.time() < deadline:
            current_url = self.evaluate("location.href") or ""
            ready = self.evaluate("document.readyState")
            current_id = parse_dataset_id_from_url(current_url)
            target_loaded = bool(target_id and current_id == target_id)
            if not target_id:
                target_loaded = normalize_space(current_url).split("#", 1)[0] == normalize_space(url).split("#", 1)[0]
            if ready in {"interactive", "complete"}:
                return
            if target_loaded and time.time() - started_at >= 1.0:
                text_len = len(normalize_space(self.evaluate("document.body ? document.body.innerText : ''") or ""))
                if text_len >= 80:
                    return
            time.sleep(0.3)

    @property
    def page_source(self) -> str:
        return self.evaluate("document.documentElement ? document.documentElement.outerHTML : ''") or ""

    def quit(self) -> None:
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass


def connect_chrome(debugger_address: str, navigate_wait_seconds: float = 20.0):
    try:
        return CDPChrome(debugger_address, navigate_wait_seconds=navigate_wait_seconds)
    except Exception as exc:
        print_chrome_help()
        raise RuntimeError(
            f"Could not connect to Chrome DevTools at {debugger_address}. "
            "Please restart Chrome with --remote-debugging-port=9222."
        ) from exc


def chrome_debugging_port(debugger_address: str) -> str:
    if ":" in debugger_address:
        return debugger_address.rsplit(":", 1)[-1]
    return "9222"


def default_chrome_path() -> str:
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def default_chrome_user_data_dir() -> str:
    return os.path.join(gettempdir(), "sh-data-browser")


def close_controlled_chrome(driver) -> None:
    try:
        if isinstance(driver, CDPChrome):
            try:
                driver.send("Browser.close")
            except Exception:
                driver.quit()
        elif hasattr(driver, "quit"):
            driver.quit()
    except Exception:
        pass


def restart_controlled_chrome(driver, args: argparse.Namespace):
    close_controlled_chrome(driver)
    time.sleep(max(0.5, getattr(args, "restart_wait", 4.0)))

    chrome_path = getattr(args, "chrome_path", "") or default_chrome_path()
    user_data_dir = getattr(args, "chrome_user_data_dir", "") or default_chrome_user_data_dir()
    port = chrome_debugging_port(getattr(args, "debugger_address", "127.0.0.1:9222"))
    url = getattr(args, "list_url", LIST_URL) or LIST_URL
    if not os.path.exists(chrome_path):
        raise RuntimeError(f"Chrome executable not found: {chrome_path}")

    cmd = [
        chrome_path,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={user_data_dir}",
        url,
    ]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(max(2.0, getattr(args, "restart_wait", 4.0)))
    return connect_chrome(
        getattr(args, "debugger_address", "127.0.0.1:9222"),
        navigate_wait_seconds=getattr(args, "navigate_wait_seconds", 20.0),
    )


def check_browser(args: argparse.Namespace) -> None:
    driver = connect_chrome(args.debugger_address, navigate_wait_seconds=getattr(args, "navigate_wait_seconds", 20.0))
    try:
        if args.open_list:
            driver.get(args.list_url)
            wait_for_page_text(driver, timeout=args.timeout)
        text = normalize_space(driver_inner_text(driver))
        print(f"[ok] connected: {args.debugger_address}")
        print(f"[ok] title: {console_safe(driver_title(driver))}")
        print(f"[ok] url: {console_safe(driver_current_url(driver))}")
        print(f"[ok] body_text_len: {len(text)}")
        if text:
            print(f"[ok] body_text_head: {console_safe(text[:300])}")
        if len(text) < 30:
            print("[warn] Browser is connected, but page text is very short. Please open/login data.sh.gov.cn.")
    finally:
        if hasattr(driver, "quit"):
            driver.quit()


def driver_inner_text(driver) -> str:
    try:
        if isinstance(driver, CDPChrome):
            return driver.evaluate("document.body ? document.body.innerText : ''") or ""
        return driver.execute_script("return document.body ? document.body.innerText : '';") or ""
    except Exception:
        return ""


def driver_page_source(driver) -> str:
    if isinstance(driver, CDPChrome):
        return driver.page_source
    return driver.page_source


def driver_evaluate(driver, expression: str):
    if isinstance(driver, CDPChrome):
        return driver.evaluate(expression)
    return driver.execute_script(f"return ({expression});")


def wait_for_page_text(driver, min_len: int = 30, timeout: float = 12.0) -> str:
    deadline = time.time() + timeout
    text = ""
    while time.time() < deadline:
        text = driver_inner_text(driver)
        if len(normalize_space(text)) >= min_len:
            return text
        time.sleep(0.5)
    return text


def driver_title(driver) -> str:
    try:
        if isinstance(driver, CDPChrome):
            return driver.evaluate("document.title") or ""
        return driver.title or ""
    except Exception:
        return ""


def driver_current_url(driver) -> str:
    try:
        if isinstance(driver, CDPChrome):
            return driver.evaluate("location.href") or ""
        return driver.current_url or ""
    except Exception:
        return ""


def scroll_full_page_browser(driver, pause: float = 0.35, max_steps: int = 18) -> None:
    for step in range(max_steps):
        try:
            y = step * 850
            if isinstance(driver, CDPChrome):
                driver.evaluate(f"window.scrollTo(0, {y}); document.body ? document.body.scrollHeight : 0")
            else:
                driver.execute_script("window.scrollTo(0, arguments[0]);", y)
        except Exception:
            break
        time.sleep(pause)
    try:
        if isinstance(driver, CDPChrome):
            driver.evaluate("window.scrollTo(0, 0)")
        else:
            driver.execute_script("window.scrollTo(0, 0);")
    except Exception:
        pass


def click_detail_modules(driver) -> int:
    """Click visible detail-page module tabs/buttons that often lazy-load fields."""
    labels = [
        "数据项",
        "字段",
        "数据样例",
        "API文档",
        "API 文档",
        "API说明",
        "数据集推荐",
        "用户评分",
        "评论",
    ]
    clicked = 0
    for label in labels:
        script = """
        (() => {
          const label = arguments[0];
          const nodes = Array.from(document.querySelectorAll('a,button,li,span,div'));
          for (const el of nodes) {
            const text = (el.innerText || el.textContent || '').trim();
            if (!text || !text.includes(label)) continue;
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            if (style.display === 'none' || style.visibility === 'hidden' || rect.width <= 0 || rect.height <= 0) continue;
            el.scrollIntoView({block: 'center', inline: 'center'});
            el.click();
            return true;
          }
          return false;
        })()
        """
        try:
            if isinstance(driver, CDPChrome):
                ok = bool(driver.evaluate(script.replace("arguments[0]", json.dumps(label, ensure_ascii=False))))
            else:
                ok = bool(driver.execute_script(script, label))
            if ok:
                clicked += 1
                time.sleep(0.45)
        except Exception:
            continue
    return clicked


def parse_dataset_id_from_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    query = parse_qs(parsed.query.replace("&&", "&"))
    return normalize_id(query.get("id", [""])[0])


def extract_formats_from_text(text: str) -> list[str]:
    found = []
    upper = normalize_space(text).upper()
    for label in FORMAT_LABELS:
        if re.search(rf"(?<![A-Z]){label}(?![A-Z])", upper):
            found.append(label)
    return sorted(set(found), key=FORMAT_LABELS.index)


def format_record(dataset_id: str, formats: list[str], source: str, status: str = "list_format") -> dict:
    record = {
        "数据集ID": normalize_id(dataset_id),
        "download_formats": ";".join(formats),
        "format_count": len(formats),
        "scrape_status": status,
        "scrape_error": "",
        "scraped_at": datetime.now().isoformat(timespec="seconds"),
        "format_source": source,
    }
    for label, col in FORMAT_COLUMNS.items():
        record[col] = int(label in formats)
    return record


def find_nearest_card_text(tag, max_depth: int = 8) -> str:
    current = tag
    best = normalize_space(tag.get_text(" ", strip=True)) if hasattr(tag, "get_text") else ""
    for _ in range(max_depth):
        if current is None:
            break
        text = normalize_space(current.get_text(" ", strip=True))
        if len(text) > len(best):
            best = text
        if any(label in text.upper() for label in FORMAT_LABELS) and len(text) > 40:
            return text
        current = current.parent
    return best


def parse_list_page_formats(html: str, master_names: dict[str, str]) -> tuple[list[dict], list[dict]]:
    soup = BeautifulSoup(html, "html.parser")
    records: dict[str, dict] = {}
    unmatched = []

    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if "detail" not in href and "id=" not in href:
            continue
        dataset_id = parse_dataset_id_from_url(href)
        if not dataset_id:
            continue
        card_text = find_nearest_card_text(a)
        formats = extract_formats_from_text(card_text)
        if formats:
            records[dataset_id] = format_record(dataset_id, formats, "list_page")

    if records:
        return list(records.values()), unmatched

    # Fallback: split visible card-like blocks and exact-match full names.
    text = normalize_space(soup.get_text("\n", strip=True))
    blocks = re.split(r"\n(?=[^\n]{4,80}(?:\n|$))", text)
    for block in blocks:
        formats = extract_formats_from_text(block)
        if not formats:
            continue
        matched_ids = [did for did, name in master_names.items() if name and name in block]
        if len(matched_ids) == 1:
            records[matched_ids[0]] = format_record(matched_ids[0], formats, "list_page_name_match")
        else:
            unmatched.append(
                {
                    "formats": ";".join(formats),
                    "match_count": len(matched_ids),
                    "text_snippet": block[:500],
                    "scraped_at": datetime.now().isoformat(timespec="seconds"),
                }
            )

    return list(records.values()), unmatched


def click_next_page(driver) -> bool:
    if isinstance(driver, CDPChrome):
        script = r"""
        (() => {
          const nodes = Array.from(document.querySelectorAll('a,button,li,span,div'));
          for (const el of nodes) {
            const text = (el.innerText || el.textContent || el.title || el.getAttribute('aria-label') || '').trim();
            const cls = String(el.className || '');
            if (!(/下一页|下页|next/i.test(text) || /next/i.test(cls))) continue;
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            const disabled = el.disabled || /disabled/.test(cls) || el.getAttribute('aria-disabled') === 'true';
            if (style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0 && !disabled) {
              el.click();
              return true;
            }
          }
          return false;
        })()
        """
        try:
            return bool(driver.evaluate(script))
        except Exception:
            return False

    from selenium.webdriver.common.by import By

    xpaths = [
        "//a[contains(normalize-space(.), '下一页')]",
        "//button[contains(normalize-space(.), '下一页')]",
        "//*[contains(@class, 'next') and not(contains(@class, 'disabled'))]",
        "//*[contains(@title, '下一页')]",
    ]
    for xp in xpaths:
        try:
            elems = driver.find_elements(By.XPATH, xp)
            for elem in elems:
                if elem.is_displayed() and elem.is_enabled():
                    driver.execute_script("arguments[0].click();", elem)
                    return True
        except Exception:
            continue
    return False


def collect_list_formats(args: argparse.Namespace) -> None:
    paths = make_paths(args)
    master = read_master(paths.master_xlsx)
    master_names = {
        normalize_id(row["数据集ID"]): normalize_space(row["数据资源名称"])
        for _, row in master[["数据集ID", "数据资源名称"]].iterrows()
    }

    driver = connect_chrome(args.debugger_address, navigate_wait_seconds=getattr(args, "navigate_wait_seconds", 20.0))
    if args.list_url:
        driver.get(args.list_url)
        wait_for_page_text(driver, timeout=args.timeout)

    all_unmatched = []
    total_found = 0
    page_no = 0
    max_pages = args.max_pages

    try:
        while True:
            page_no += 1
            wait_for_page_text(driver, timeout=args.timeout)
            html = driver_page_source(driver)
            records, unmatched = parse_list_page_formats(html, master_names)
            total_found += len(records)
            all_unmatched.extend(unmatched)
            if records:
                upsert_checkpoint(paths.checkpoint_csv, records)
                sync_master_from_checkpoint(paths)
            print(f"[list] page={page_no} matched={len(records)} unmatched={len(unmatched)}")

            if max_pages and page_no >= max_pages:
                break
            if not click_next_page(driver):
                break
            time.sleep(args.delay)
    finally:
        if all_unmatched:
            pd.DataFrame(all_unmatched).to_csv(paths.list_unmatched_csv, index=False, encoding="utf-8-sig")

    print(f"[ok] list format records saved/updated: {total_found}")
    print(f"[ok] checkpoint: {paths.checkpoint_csv}")


def metadata_from_tables(soup: BeautifulSoup) -> dict[str, str]:
    result: dict[str, str] = {}
    cells = soup.find_all(["td", "th"])
    for idx, cell in enumerate(cells[:-1]):
        label = to_simplified_text(cell.get_text(" ", strip=True)).rstrip(":：")
        if not label or len(label) > 30:
            continue
        value = to_simplified_text(cells[idx + 1].get_text(" ", strip=True))
        if value and label not in result:
            result[label] = value
    return result


def regex_value(text: str, labels: list[str], max_chars: int = 80) -> str:
    for label in labels:
        pattern = rf"{re.escape(label)}\s*[:：]?\s*([^\n\r]{{1,{max_chars}}})"
        match = re.search(pattern, text)
        if match:
            value = normalize_space(match.group(1))
            value = re.split(r"\s{2,}| 数据标签| 关键词| 更新| 开放| 空间| 时间", value)[0]
            return normalize_space(value)
    return ""


def extract_tables(soup: BeautifulSoup) -> list[list[list[str]]]:
    tables = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [normalize_space(c.get_text(" ", strip=True)) for c in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables


def looks_like_field_name(value: str) -> bool:
    value = to_simplified_text(value)
    lower_value = value.lower()
    if lower_value in {"nan", "none", "null"} or lower_value.startswith("unnamed"):
        return False
    if value in FIELD_NAME_BAD_VALUES:
        return False
    if value.isdigit():
        return False
    if any(token in value for token in ["标签", "標籤", "关键", "關鍵", "下载", "下載"]):
        return False
    if len(value) > 30:
        return False
    if re.fullmatch(r"\d{4}[./-]\d{1,2}(?:[./-]\d{1,2})?", value):
        return False
    if re.fullmatch(r"[\d./:： -]+", value):
        return False
    if re.search(r"\d+(?:米|号|號|年|月|日|:|：)", value):
        return False
    if ("路" in value and any(token in value for token in ["约", "進", "进", "出", "侧", "側", "段", "隧道"])):
        return False
    if is_weak_chinese_field_label(value) and len(value) > 12:
        return False
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$|^[\u4e00-\u9fffA-Za-z0-9_（）()、/.-]+$", value))


def is_weak_chinese_field_label(value: str) -> bool:
    if not re.search(r"[\u4e00-\u9fff]", value):
        return False
    if re.search(r"[A-Za-z_]", value):
        return False
    return not any(hint in value for hint in FIELD_NAME_HINTS)


def clean_field_names(fields: Iterable[object], drop_weak: bool = True) -> list[str]:
    cleaned = []
    for field in fields:
        name = to_simplified_text(field)
        if looks_like_field_name(name):
            cleaned.append(name)
    cleaned = sorted(set(cleaned))
    weak = [name for name in cleaned if is_weak_chinese_field_label(name)]
    if drop_weak and len(cleaned) >= 4 and len(weak) >= max(3, math.ceil(len(cleaned) * 0.5)):
        cleaned = [name for name in cleaned if name not in weak]
    return cleaned


def clean_field_names_text(value: object) -> str:
    parts = re.split(r"[;；,，\n\r]+", normalize_space(value))
    return ";".join(clean_field_names(parts))


def extract_field_info(tables: list[list[list[str]]]) -> tuple[list[str], int, int, list[str]]:
    field_names: list[str] = []
    description_count = 0
    sample_headers: list[str] = []
    pending_schema: tuple[int, int | None] | None = None

    for rows in tables:
        if not rows:
            continue
        header = [to_simplified_text(c) for c in rows[0]]
        header_joined = " ".join(header)
        name_idx = None
        desc_idx = None
        strong_name_keys = [
            "字段名",
            "字段名称",
            "字段英文名",
            "字段代码",
            "字段编码",
            "字段标识",
            "栏位名",
            "栏位名称",
            "参数名",
            "参数名称",
            "name",
            "Name",
        ]
        description_keys = [
            "描述",
            "说明",
            "含义",
            "备注",
            "字段描述",
            "栏位描述",
            "description",
            "Description",
        ]
        for idx, col in enumerate(header):
            if any(key in col for key in strong_name_keys):
                name_idx = idx
                break
        if name_idx is None and any(key in header_joined for key in ["字段", "栏位", "参数", "描述", "说明"]):
            for idx, col in enumerate(header):
                if col in {"名称", "参数", "name", "Name"}:
                    name_idx = idx
                    break
        for idx, col in enumerate(header):
            if any(key in col for key in description_keys):
                desc_idx = idx
                break

        if pending_schema is not None:
            pending_name_idx, pending_desc_idx = pending_schema
            parsed_pending = False
            for row in rows:
                if not row or pending_name_idx >= len(row):
                    continue
                first_cell = normalize_space(row[0])
                if not first_cell.isdigit():
                    continue
                name = normalize_space(row[pending_name_idx])
                if looks_like_field_name(name):
                    field_names.append(name)
                    if (
                        pending_desc_idx is not None
                        and pending_desc_idx < len(row)
                        and normalize_space(row[pending_desc_idx])
                    ):
                        description_count += 1
                    parsed_pending = True
            pending_schema = None
            if parsed_pending:
                continue

        if name_idx is not None and any(key in header_joined for key in ["字段", "栏位", "参数", "名称", "描述", "说明"]):
            if len(rows) == 1:
                pending_schema = (name_idx, desc_idx)
                continue
            for row in rows[1:]:
                if name_idx >= len(row):
                    continue
                name = normalize_space(row[name_idx])
                if looks_like_field_name(name):
                    field_names.append(name)
                    if desc_idx is not None and desc_idx < len(row) and normalize_space(row[desc_idx]):
                        description_count += 1
            continue

        if len(header) >= 2 and not any(key in header_joined for key in ["数据标签", "关键字", "更新频率", "开放条件"]):
            plausible = [h for h in header if looks_like_field_name(h)]
            if len(plausible) >= 2:
                sample_headers.extend(plausible)

    field_names = clean_field_names(field_names, drop_weak=False)
    sample_headers = clean_field_names(sample_headers)
    if not field_names and sample_headers:
        field_names = sample_headers

    return field_names, len(field_names), description_count, sample_headers


def extract_recommendations(text: str) -> tuple[int, str]:
    text = to_simplified_text(text)
    if "数据集推荐" not in text:
        return 0, ""
    after = text.split("数据集推荐", 1)[-1][:5000]
    lines = [normalize_space(x) for x in after.splitlines() if normalize_space(x)]
    names = []
    for line in lines:
        if any(skip in line for skip in ["了解详情", "查看", "调用", "浏览", "下载", "评分"]):
            continue
        if 4 <= len(line) <= 80 and not re.search(r"^\d+$", line):
            names.append(line)
    names = names[:20]
    count = max(after.count("了解详情"), len(names))
    return count, ";".join(names)


def extract_rating_and_comments(text: str) -> tuple[float | None, int | None]:
    text = to_simplified_text(text)
    rating = None
    comment_count = None
    match = re.search(r"用户评分[^0-9]*(\d+(?:\.\d+)?)", text)
    if match:
        rating = float(match.group(1))
    match = re.search(r"评论[^0-9]*(\d+)", text)
    if match:
        comment_count = int(match.group(1))
    return rating, comment_count


def detect_api_need_apply(text: str, metadata: dict[str, str]) -> int:
    joined = to_simplified_text(text + " " + " ".join(metadata.values()))
    need_words = ["申请使用", "依申请", "有条件开放", "审核后开放", "请先申请", "需要申请"]
    open_words = ["无条件开放", "直接下载", "无需申请"]
    if any(word in joined for word in need_words):
        return 1
    if any(word in joined for word in open_words):
        return 0
    return 0


def infer_field_flags(field_names: list[str], text: str) -> tuple[int, int]:
    joined = to_simplified_text(" ".join(field_names) + " " + normalize_space(text)).lower()
    time_patterns = [
        "时间",
        "時間",
        "日期",
        "年份",
        "年度",
        "年月",
        "time",
        "date",
        "year",
        "update_time",
        "jpt_update_time",
    ]
    geo_patterns = [
        "地址",
        "位置",
        "经纬度",
        "經緯度",
        "经度",
        "經度",
        "纬度",
        "緯度",
        "坐标",
        "坐標",
        "街道",
        "街镇",
        "街鎮",
        "区县",
        "空间",
        "location",
        "lng",
        "lat",
        "longitude",
        "latitude",
    ]
    return int(any(p.lower() in joined for p in time_patterns)), int(any(p.lower() in joined for p in geo_patterns))


def build_detail_remark(record: dict, text: str) -> str:
    notes = []
    if is_missing_detail_value(record.get("download_formats")):
        notes.append("未识别到格式标签")
    if is_missing_detail_value(record.get("field_names")):
        notes.append("未识别到字段列表")
    if is_missing_detail_value(record.get("record_count")):
        notes.append("未识别到数据量")
    if is_missing_detail_value(record.get("data_size")):
        notes.append("未识别到数据大小")
    if "验证码" in text or "访问过于频繁" in text:
        notes.append("页面疑似出现访问限制")
    return "；".join(notes)


def parse_detail_html(html: str, text: str, dataset_id: str) -> dict:
    text = to_simplified_text(text)
    soup = BeautifulSoup(html, "html.parser")
    metadata = metadata_from_tables(soup)
    tables = extract_tables(soup)
    field_names, field_count, desc_count, sample_headers = extract_field_info(tables)
    formats = extract_formats_from_text(text)
    rec_count, rec_names = extract_recommendations(text)
    rating, comment_count = extract_rating_and_comments(text)
    has_time_field, has_geo_field = infer_field_flags(field_names, text)

    record_count_text = (
        metadata.get("数据量（条）")
        or metadata.get("数据量(条)")
        or metadata.get("数据量 条")
        or metadata.get("数据量")
        or regex_value(text, ["数据量（条）", "数据量(条)", "数据量"])
    )
    record_count_value = safe_int(record_count_text)
    if record_count_value is None and normalize_space(record_count_text) in {"-", "--", "无", "暂无"}:
        record_count_value = normalize_space(record_count_text)
    data_size = to_simplified_text(metadata.get("数据大小") or regex_value(text, ["数据大小"]))
    spatial_scope = to_simplified_text(metadata.get("空间范围") or regex_value(text, ["空间范围"]))
    time_scope = to_simplified_text(metadata.get("时间范围") or regex_value(text, ["时间范围"]))

    record = {
        "数据集ID": normalize_id(dataset_id),
        "download_formats": ";".join(formats),
        "api_need_apply": detect_api_need_apply(text, metadata),
        "format_count": len(formats),
        "record_count": record_count_value,
        "data_size": data_size,
        "field_names": ";".join(field_names),
        "field_count": field_count,
        "has_time_field": has_time_field,
        "has_geo_field": has_geo_field,
        "field_description_count": desc_count,
        "has_standard_field_description": int(desc_count > 0),
        "has_data_sample": int("数据样例" in text or "DATA SAMPLE" in text or bool(sample_headers)),
        "sample_field_headers": ";".join(sample_headers),
        "recommended_dataset_count": rec_count,
        "recommended_dataset_names": rec_names,
        "rating_score": rating,
        "comment_count": comment_count,
        "detail_spatial_scope": spatial_scope,
        "detail_time_scope": time_scope,
        "scrape_status": "success",
        "scrape_error": "",
        "scraped_at": datetime.now().isoformat(timespec="seconds"),
    }
    for label, col in FORMAT_COLUMNS.items():
        record[col] = int(label in formats)
    record["备注"] = build_detail_remark(record, text)
    return record


def is_missing_detail_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = normalize_space(value)
    return text == "" or text.lower() in {"nan", "none", "null"}


def missing_core_detail_fields(record: dict) -> list[str]:
    missing = []
    for key in ["download_formats", "record_count", "data_size", "field_names"]:
        value = record.get(key)
        if is_missing_detail_value(value):
            missing.append(key)
    return missing


def parse_detail_with_retries(driver, url: str, dataset_id: str, args: argparse.Namespace) -> dict:
    last_record = None
    max_attempts = max(1, getattr(args, "detail_max_attempts", 3))
    for attempt in range(max_attempts):
        if attempt == 0:
            driver.get(url)
        elif attempt == 1:
            missing = missing_core_detail_fields(last_record or {})
            print(f"[retry] {dataset_id} targeted retry for: {','.join(missing)}", flush=True)
            time.sleep(0.8)
            scroll_full_page_browser(driver, pause=0.18, max_steps=12)
            click_detail_modules(driver)
        else:
            missing = missing_core_detail_fields(last_record or {})
            print(f"[retry] {dataset_id} reload retry for: {','.join(missing)}", flush=True)
            driver.get(url)
            wait_for_page_text(driver, min_len=30, timeout=args.timeout)

        initial_text = wait_for_page_text(driver, min_len=30, timeout=args.timeout)
        if is_unavailable_detail_page(initial_text):
            return unavailable_detail_record(dataset_id, url, initial_text)
        scroll_steps = 12 if attempt == 0 else 16
        scroll_full_page_browser(driver, pause=0.16, max_steps=scroll_steps)
        clicked_modules = click_detail_modules(driver)
        if clicked_modules:
            time.sleep(0.45 if attempt == 0 else 0.75)
            scroll_full_page_browser(driver, pause=0.12, max_steps=8 if attempt == 0 else 12)

        text = wait_for_page_text(driver, min_len=args.min_text_len, timeout=args.timeout)
        if is_unavailable_detail_page(text):
            return unavailable_detail_record(dataset_id, url, text)
        html = driver_page_source(driver)
        if len(normalize_space(text)) < args.min_text_len:
            raise RuntimeError(f"empty or blocked page text_len={len(normalize_space(text))}")

        record = parse_detail_html(html, text, dataset_id)
        record["detail_url"] = url
        last_record = record
        if not missing_core_detail_fields(record):
            break

    if last_record is None:
        raise RuntimeError("detail page parsing produced no record")
    missing_core = missing_core_detail_fields(last_record)
    if missing_core:
        last_record["备注"] = "；".join(
            part for part in [normalize_space(last_record.get("备注")), f"核心详情字段仍缺失：{','.join(missing_core)}"] if part
        )
    return last_record


def collect_detail_fields(args: argparse.Namespace) -> None:
    paths = make_paths(args)
    master = read_master(paths.master_xlsx)
    checkpoint = read_checkpoint(paths.checkpoint_csv)
    done_ids = set()
    if not args.force and not checkpoint.empty and "scrape_status" in checkpoint.columns:
        terminal = checkpoint["scrape_status"].map(is_terminal_scrape_status)
        done_ids = set(checkpoint.loc[terminal, "数据集ID"].map(normalize_id))

    tasks = master.copy()
    tasks["数据集ID"] = tasks["数据集ID"].map(normalize_id)
    if done_ids:
        tasks = tasks[~tasks["数据集ID"].isin(done_ids)]
    if args.start_index:
        tasks = tasks.iloc[args.start_index :]
    if args.limit:
        tasks = tasks.head(args.limit)

    driver = connect_chrome(args.debugger_address, navigate_wait_seconds=getattr(args, "navigate_wait_seconds", 20.0))
    failed_records = []
    saved_since_sync = 0
    recent_durations: list[float] = []
    last_restart_at = time.time()
    restart_count = 0

    for row_no, (_, row) in enumerate(tqdm(tasks.iterrows(), total=len(tasks), desc="detail pages"), start=1):
        row_started_at = time.time()
        dataset_id = normalize_id(row["数据集ID"])
        url = normalize_space(row.get("detail_url", "")) or build_detail_url(dataset_id, normalize_space(row.get("type_code", "")))
        if not url:
            fail = {
                "数据集ID": dataset_id,
                "detail_url": "",
                "scrape_status": "failed",
                "scrape_error": "missing detail_url",
                "scraped_at": datetime.now().isoformat(timespec="seconds"),
            }
            failed_records.append(fail)
            upsert_checkpoint(paths.checkpoint_csv, [fail])
            saved_since_sync += 1
            if saved_since_sync >= args.sync_every:
                sync_master_from_checkpoint(paths)
                saved_since_sync = 0
            continue

        try:
            record = parse_detail_with_retries(driver, url, dataset_id, args)
            upsert_checkpoint(paths.checkpoint_csv, [record])
            saved_since_sync += 1
        except Exception as exc:
            fail = {
                "数据集ID": dataset_id,
                "detail_url": url,
                "scrape_status": "failed",
                "scrape_error": repr(exc),
                "scraped_at": datetime.now().isoformat(timespec="seconds"),
                "备注": "页面打不开、空白或疑似被拦截，待重试",
            }
            failed_records.append(fail)
            upsert_checkpoint(paths.checkpoint_csv, [fail])
            saved_since_sync += 1
            if args.stop_on_block and "empty or blocked" in repr(exc):
                print(f"[stop] possible anti-crawl block at {dataset_id}: {exc}")
                break

        if saved_since_sync >= args.sync_every:
            sync_master_from_checkpoint(paths)
            saved_since_sync = 0

        row_duration = time.time() - row_started_at
        recent_durations.append(row_duration)
        window = max(1, getattr(args, "restart_window", 100))
        if len(recent_durations) > window:
            recent_durations = recent_durations[-window:]
        should_check_restart = (
            getattr(args, "auto_restart_on_slow", False)
            and len(recent_durations) >= window
            and row_no < len(tasks)
        )
        if should_check_restart:
            avg_duration = sum(recent_durations) / len(recent_durations)
            restart_allowed_by_interval = time.time() - last_restart_at >= getattr(args, "min_restart_interval", 600.0)
            max_restarts = getattr(args, "max_restarts", 0)
            restart_allowed_by_count = max_restarts <= 0 or restart_count < max_restarts
            if (
                avg_duration > getattr(args, "slow_threshold_seconds", 25.0)
                and restart_allowed_by_interval
                and restart_allowed_by_count
            ):
                print(
                    f"[restart] avg_last_{window}={avg_duration:.1f}s > "
                    f"{getattr(args, 'slow_threshold_seconds', 25.0):.1f}s; restarting Chrome",
                    flush=True,
                )
                sync_master_from_checkpoint(paths)
                driver = restart_controlled_chrome(driver, args)
                restart_count += 1
                last_restart_at = time.time()
                recent_durations.clear()
        time.sleep(args.delay)

    sync_master_from_checkpoint(paths)
    all_failed = pd.DataFrame(failed_records)
    if not all_failed.empty:
        all_failed.to_csv(paths.failed_csv, index=False, encoding="utf-8-sig")
    print(f"[ok] checkpoint: {paths.checkpoint_csv}")
    print(f"[ok] master synced: {paths.master_xlsx}")
    if failed_records:
        print(f"[warn] failed records in this run: {len(failed_records)} -> {paths.failed_csv}")


def collect_web_fields(args: argparse.Namespace) -> None:
    if getattr(args, "list_pages", 0) > 0:
        args.max_pages = args.list_pages
        print(f"[phase] collect list formats: pages={args.max_pages}")
        collect_list_formats(args)
    else:
        print("[phase] skip list-page format sweep; detail pages will still parse formats.")

    print("[phase] collect detail fields")
    collect_detail_fields(args)

    paths = make_paths(args)
    master = sync_master_from_checkpoint(paths)
    min_df = export_minimum_required(paths, master)
    print(f"[ok] minimum rows={len(min_df)}")
    print(f"[ok] minimum table: {paths.minimum_required_xlsx}")


def checkpoint_success_count(path: Path) -> int:
    checkpoint = read_checkpoint(path)
    if checkpoint.empty or "scrape_status" not in checkpoint.columns:
        return 0
    return int((checkpoint["scrape_status"].astype(str) == "success").sum())


def run_web_batches(args: argparse.Namespace) -> None:
    paths = make_paths(args)
    batch_no = 0
    while True:
        if args.max_batches and batch_no >= args.max_batches:
            print(f"[done] reached max_batches={args.max_batches}")
            break

        before = checkpoint_success_count(paths.checkpoint_csv)
        remaining = max(0, len(read_master(paths.master_xlsx)) - before)
        if remaining <= 0:
            print("[done] all rows already have successful web supplement records.")
            break

        batch_no += 1
        args.limit = min(args.batch_size, remaining)
        args.start_index = 0
        print(f"[batch] {batch_no} start; before_success={before}; target={args.limit}; remaining={remaining}")

        collect_detail_fields(args)
        master = sync_master_from_checkpoint(paths)
        export_minimum_required(paths, master)

        args.last = args.limit
        inspect_minimum(args)

        after = checkpoint_success_count(paths.checkpoint_csv)
        print(f"[batch] {batch_no} done; after_success={after}; new_success={after - before}")
        if after <= before:
            print("[stop] no new successful records were added in this batch.")
            break
        if args.sleep_between_batches:
            time.sleep(args.sleep_between_batches)


def refresh_outputs(args: argparse.Namespace) -> None:
    paths = make_paths(args)
    master = sync_master_from_checkpoint(paths)
    min_df = export_minimum_required(paths, master)
    print(f"[ok] master rows={len(master)}: {paths.master_xlsx}")
    print(f"[ok] minimum rows={len(min_df)}: {paths.minimum_required_xlsx}")


def has_mojibake(value: object) -> bool:
    text = normalize_space(value)
    if not text:
        return False
    return bool(re.search(r"�|ï¿½|æ.|å.|ç.|è.|é.|鏁|鍙|涓|浠|鐨|绫|閲|鎴", text))


def has_noise_field(value: object) -> bool:
    text = normalize_space(value)
    if not text:
        return False
    noise = [
        "下载格式",
        "全选",
        "字段描述",
        "字段类型",
        "字段大小",
        "参数描述",
        "参数类型",
        "数据标签",
        "栏位名称",
        "栏位长度",
        "字元型C",
        "字符型",
        "数字型",
        "檔案名稱",
        "關鍵字",
    ]
    return any(token in text for token in noise)


def has_suspicious_field_names(value: object) -> bool:
    names = [name for name in re.split(r"[;；,，\n\r]+", normalize_space(value)) if name]
    if not names:
        return False
    if any(re.fullmatch(r"\d{4}[./-]\d{1,2}(?:[./-]\d{1,2})?", name) for name in names):
        return True
    weak = [name for name in names if is_weak_chinese_field_label(to_simplified_text(name))]
    return len(names) >= 4 and len(weak) >= max(3, math.ceil(len(names) * 0.5))


def has_traditional_residue(value: object) -> bool:
    text = normalize_space(value)
    if not text:
        return False
    return any(chr(codepoint) in text for codepoint, converted in TRADITIONAL_TO_SIMPLIFIED.items() if chr(codepoint) != converted)


def inspect_minimum(args: argparse.Namespace) -> None:
    paths = make_paths(args)
    if not paths.minimum_required_xlsx.exists():
        export_minimum_required(paths)
    minimum = pd.read_excel(paths.minimum_required_xlsx, dtype={"数据集ID": str})
    checkpoint = read_checkpoint(paths.checkpoint_csv)
    if checkpoint.empty or "scrape_status" not in checkpoint.columns:
        sample = minimum.head(0)
    else:
        ok = checkpoint[checkpoint["scrape_status"].astype(str) == "success"].copy()
        if "scraped_at" in ok.columns:
            ok = ok.sort_values("scraped_at")
        ids = ok["数据集ID"].map(normalize_id).tail(args.last).tolist()
        sample = minimum[minimum["数据集ID"].map(normalize_id).isin(ids)].copy()

    text_cols = [c for c in ["数据资源名称", "field_names", "data_size", "spatial_scope", "time_scope", "备注"] if c in sample.columns]
    report = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "rows_checked": int(len(sample)),
        "missing_download_formats": int(sample.get("download_formats", pd.Series(dtype=object)).isna().sum()) if len(sample) else 0,
        "missing_field_names": int((sample.get("field_names", pd.Series(dtype=object)).fillna("").astype(str) == "").sum()) if len(sample) else 0,
        "zero_field_count": int((pd.to_numeric(sample.get("field_count", pd.Series(dtype=object)), errors="coerce").fillna(0) <= 0).sum()) if len(sample) else 0,
        "mojibake_rows": int(sample[text_cols].apply(lambda row: any(has_mojibake(v) for v in row), axis=1).sum()) if len(sample) and text_cols else 0,
        "traditional_residue_rows": int(sample[text_cols].apply(lambda row: any(has_traditional_residue(v) for v in row), axis=1).sum()) if len(sample) and text_cols else 0,
        "noise_field_rows": int(sample.get("field_names", pd.Series(dtype=object)).map(has_noise_field).sum()) if len(sample) else 0,
        "suspicious_field_rows": int(sample.get("field_names", pd.Series(dtype=object)).map(has_suspicious_field_names).sum()) if len(sample) else 0,
    }
    report_path = paths.output_dir / "batch_quality_report.csv"
    old = pd.read_csv(report_path, encoding="utf-8-sig") if report_path.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([report])], ignore_index=True).to_csv(report_path, index=False, encoding="utf-8-sig")

    print("[quality]", json.dumps(report, ensure_ascii=False))
    print(f"[quality] report: {report_path}")
    if report["mojibake_rows"] or report["noise_field_rows"] or report["suspicious_field_rows"]:
        raise RuntimeError("Quality check found mojibake or noisy field names. Stop and inspect before continuing.")


def import_sklearn():
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import KFold, cross_val_predict
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for fit-model. Install with:\n"
            "  .venv/bin/python -m pip install -r 'Statistics Collect/requirements.txt'"
        ) from exc
    return {
        "ColumnTransformer": ColumnTransformer,
        "TfidfVectorizer": TfidfVectorizer,
        "SimpleImputer": SimpleImputer,
        "Ridge": Ridge,
        "mean_absolute_error": mean_absolute_error,
        "r2_score": r2_score,
        "KFold": KFold,
        "cross_val_predict": cross_val_predict,
        "Pipeline": Pipeline,
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
    }


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "数据资源名称",
        "数据资源内容描述",
        "数据资源提供部门",
        "数据领域",
        "开放条件",
        "数据资源类型",
        "数据资源状态",
        "更新频率",
        "开放属性",
        "spatial_admin_level",
        "type_code",
        "download_formats",
    ]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)

    out["model_text"] = (
        out["数据资源名称"]
        + " "
        + out["数据资源内容描述"]
        + " "
        + out["数据领域"]
        + " "
        + out["开放条件"]
        + " "
        + out["download_formats"]
    )

    numeric_cols = model_numeric_columns()
    for col in numeric_cols:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().all():
            out[col] = 0

    if "ActualUse_v0" not in out.columns or out["ActualUse_v0"].isna().all():
        for col in ["浏览量", "下载量", "接口调用量"]:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
            out[f"{col}_percentile"] = percentile_rank(np.log1p(out[col]))
        out["ActualUse_v0"] = (
            0.45 * out["浏览量_percentile"]
            + 0.35 * out["下载量_percentile"]
            + 0.20 * out["接口调用量_percentile"]
        )
    out["ActualUse_percentile"] = percentile_rank(out["ActualUse_v0"])
    return out


def model_numeric_columns() -> list[str]:
    return [
        "publication_age_days",
        "update_recency_days",
        "maintenance_span_days",
        "title_len",
        "description_len",
        "description_has_field_hint",
        "scene_keyword_count",
        "has_rdf",
        "has_xml",
        "has_csv",
        "has_json",
        "has_xlsx",
        "api_need_apply",
        "format_count",
        "record_count",
        "field_count",
        "has_time_field",
        "has_geo_field",
        "field_description_count",
        "has_standard_field_description",
        "has_data_sample",
        "recommended_dataset_count",
        "comment_count",
        "rating_score",
    ]


def model_categorical_columns() -> list[str]:
    return [
        "数据资源提供部门",
        "数据领域",
        "数据资源类型",
        "数据资源状态",
        "更新频率",
        "开放属性",
        "spatial_admin_level",
        "type_code",
    ]


def classify_quadrant(potential_pct: float, actual_pct: float) -> str:
    if pd.isna(potential_pct) or pd.isna(actual_pct):
        return "未分类"
    pot_high = potential_pct >= 0.5
    actual_high = actual_pct >= 0.5
    if pot_high and actual_high:
        return "高潜高用"
    if pot_high and not actual_high:
        return "高潜低用"
    if not pot_high and actual_high:
        return "低潜高用"
    return "低潜低用"


def fit_model(args: argparse.Namespace) -> None:
    paths = make_paths(args)
    df = sync_master_from_checkpoint(paths)
    if args.limit:
        df = df.head(args.limit).copy()
    frame = prepare_model_frame(df)
    sk = import_sklearn()

    feature_cols = ["model_text"] + model_categorical_columns() + model_numeric_columns()
    X = frame[feature_cols]
    y = pd.to_numeric(frame["ActualUse_v0"], errors="coerce").fillna(0)

    text_transformer = sk["TfidfVectorizer"](analyzer="char_wb", ngram_range=(2, 4), max_features=args.max_text_features)
    categorical_transformer = sk["OneHotEncoder"](handle_unknown="ignore")
    numeric_transformer = sk["Pipeline"](
        [
            ("imputer", sk["SimpleImputer"](strategy="constant", fill_value=0, keep_empty_features=True)),
            ("scaler", sk["StandardScaler"]()),
        ]
    )
    preprocessor = sk["ColumnTransformer"](
        [
            ("text", text_transformer, "model_text"),
            ("cat", categorical_transformer, model_categorical_columns()),
            ("num", numeric_transformer, model_numeric_columns()),
        ],
        remainder="drop",
    )
    model = sk["Pipeline"](
        [
            ("preprocess", preprocessor),
            ("ridge", sk["Ridge"](alpha=args.alpha)),
        ]
    )

    n_splits = min(args.folds, len(frame))
    if n_splits < 2:
        raise ValueError("Need at least 2 rows to fit model.")
    cv = sk["KFold"](n_splits=n_splits, shuffle=True, random_state=args.random_state)
    pred = sk["cross_val_predict"](model, X, y, cv=cv)
    frame["PotentialUse_ml"] = pred
    frame["PotentialUse_ml_percentile"] = percentile_rank(frame["PotentialUse_ml"])
    frame["DormantScore"] = frame["PotentialUse_ml_percentile"] - frame["ActualUse_percentile"]
    frame["DormantRank"] = frame["DormantScore"].rank(method="first", ascending=False).astype(int)
    frame["dormant_top100"] = (frame["DormantRank"] <= 100).astype(int)
    frame["dormant_top300"] = (frame["DormantRank"] <= 300).astype(int)
    frame["quadrant"] = [
        classify_quadrant(p, a)
        for p, a in zip(frame["PotentialUse_ml_percentile"], frame["ActualUse_percentile"])
    ]
    frame["dormant_candidate"] = (
        (frame["PotentialUse_ml_percentile"] >= args.candidate_potential_threshold)
        & (frame["ActualUse_percentile"] <= args.candidate_actual_threshold)
    ).astype(int)

    metrics = {
        "rows": int(len(frame)),
        "folds": int(n_splits),
        "r2_oof": float(sk["r2_score"](y, pred)),
        "mae_oof": float(sk["mean_absolute_error"](y, pred)),
        "candidate_potential_threshold": args.candidate_potential_threshold,
        "candidate_actual_threshold": args.candidate_actual_threshold,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    paths.model_metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    frame.to_csv(paths.model_results_csv, index=False, encoding="utf-8-sig")
    export_workbook_from_frames(paths, frame, metrics)
    print(f"[ok] model rows={len(frame)} r2_oof={metrics['r2_oof']:.4f} mae_oof={metrics['mae_oof']:.4f}")
    print(f"[ok] model results: {paths.model_results_csv}")
    print(f"[ok] workbook: {paths.dormant_workbook}")


def field_dictionary() -> pd.DataFrame:
    rows = [
        ("detail_url", "由数据资源类型与数据集ID自动构造的详情页 URL"),
        ("spatial_admin_level", "由数据资源提供部门推断的行政供给层级，不等同于详情页空间范围"),
        ("publication_age_days", "首次发布日期距运行日天数"),
        ("update_recency_days", "最近更新日期距运行日天数"),
        ("maintenance_span_days", "最近更新日期与首次发布日期间隔天数"),
        ("ActualUse_v0", "浏览、下载、接口调用量 log1p 后百分位加权得分"),
        ("PotentialUse_ml", "基于非使用量字段预测的目录/网页增强潜在使用表现"),
        ("DormantScore", "PotentialUse_ml_percentile - ActualUse_percentile"),
        ("quadrant", "按潜力/实际使用中位数划分的四象限类型"),
        ("dormant_candidate", "高潜低用候选：潜力百分位高且实际使用百分位低"),
    ]
    return pd.DataFrame(rows, columns=["field", "description"])


def export_workbook_from_frames(paths: PipelinePaths, results: pd.DataFrame, metrics: dict) -> None:
    candidate_cols = [
        "数据集ID",
        "数据资源名称",
        "数据资源提供部门",
        "数据领域",
        "数据资源类型",
        "开放属性",
        "浏览量",
        "下载量",
        "接口调用量",
        "ActualUse_v0",
        "ActualUse_percentile",
        "PotentialUse_ml",
        "PotentialUse_ml_percentile",
        "DormantScore",
        "DormantRank",
        "dormant_top100",
        "dormant_top300",
        "quadrant",
        "dormant_candidate",
        "detail_url",
        "download_formats",
        "api_need_apply",
        "field_count",
        "has_time_field",
        "has_geo_field",
        "has_data_sample",
        "recommended_dataset_count",
        "record_count",
        "data_size",
        "备注",
    ]
    candidate_cols = [c for c in candidate_cols if c in results.columns]
    candidates = results.sort_values("DormantScore", ascending=False)
    quadrant_summary = (
        results.groupby("quadrant", dropna=False)
        .agg(
            resource_count=("数据集ID", "count"),
            mean_actual_use=("ActualUse_v0", "mean"),
            mean_potential=("PotentialUse_ml", "mean"),
            mean_dormant_score=("DormantScore", "mean"),
        )
        .reset_index()
        .sort_values("resource_count", ascending=False)
    )
    metrics_df = pd.DataFrame([metrics])
    failed = read_checkpoint(paths.checkpoint_csv)
    if "scrape_status" in failed.columns:
        failed = failed[failed["scrape_status"].astype(str) == "failed"]
    else:
        failed = pd.DataFrame()

    with pd.ExcelWriter(paths.dormant_workbook, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, index=False, sheet_name="model_metrics")
        quadrant_summary.to_excel(writer, index=False, sheet_name="quadrant_summary")
        candidates[candidate_cols].head(500).to_excel(writer, index=False, sheet_name="top_dormant_candidates")
        results[candidate_cols].to_excel(writer, index=False, sheet_name="model_results")
        field_dictionary().to_excel(writer, index=False, sheet_name="field_dictionary")
        if not failed.empty:
            failed.to_excel(writer, index=False, sheet_name="failed_pages")


def export_workbook(args: argparse.Namespace) -> None:
    paths = make_paths(args)
    if not paths.model_results_csv.exists():
        raise FileNotFoundError(f"Model results not found: {paths.model_results_csv}. Run fit-model first.")
    results = pd.read_csv(paths.model_results_csv, dtype={"数据集ID": str}, encoding="utf-8-sig")
    metrics = {}
    if paths.model_metrics_json.exists():
        metrics = json.loads(paths.model_metrics_json.read_text(encoding="utf-8"))
    export_workbook_from_frames(paths, results, metrics)
    if paths.master_xlsx.exists():
        export_minimum_required(paths)
    print(f"[ok] workbook: {paths.dormant_workbook}")
    print(f"[ok] minimum table: {paths.minimum_required_xlsx}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", default=DEFAULT_CATALOG, help="目录 Excel 文件路径")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出目录")


def add_detail_collection_tuning_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--detail-max-attempts", type=int, default=3, help="maximum parse attempts per detail page")
    parser.add_argument("--navigate-wait-seconds", type=float, default=20.0, help="maximum seconds to wait for Chrome navigation before detail-page readiness checks")
    parser.add_argument("--auto-restart-on-slow", action="store_true", help="restart controlled Chrome when recent pages slow down")
    parser.add_argument("--restart-window", type=int, default=100, help="number of recent pages used for slow-restart average")
    parser.add_argument("--slow-threshold-seconds", type=float, default=25.0, help="average seconds per page that triggers restart")
    parser.add_argument("--min-restart-interval", type=float, default=600.0, help="minimum seconds between automatic restarts")
    parser.add_argument("--max-restarts", type=int, default=0, help="maximum automatic restarts; 0 means unlimited")
    parser.add_argument("--restart-wait", type=float, default=4.0, help="seconds to wait around Chrome restart")
    parser.add_argument("--chrome-path", default="", help="Chrome executable path for automatic restart")
    parser.add_argument("--chrome-user-data-dir", default="", help="Chrome user-data-dir for automatic restart")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="上海公共数据目录增强与沉睡资产识别管线")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-master", help="读取目录 Excel，生成 URL 与派生字段")
    add_common_args(p)
    p.set_defaults(func=build_master)

    p = sub.add_parser("check-browser", help="检查 Chrome 调试端口是否可连接")
    add_common_args(p)
    p.add_argument("--debugger-address", default="127.0.0.1:9222")
    p.add_argument("--list-url", default=LIST_URL)
    p.add_argument("--timeout", type=float, default=12.0)
    p.add_argument("--open-list", action="store_true", help="连接后打开数据资源列表页")
    p.set_defaults(func=check_browser)

    p = sub.add_parser("collect-list-formats", help="连接已登录 Chrome，从列表页补格式标签")
    add_common_args(p)
    p.add_argument("--debugger-address", default="127.0.0.1:9222")
    p.add_argument("--list-url", default=LIST_URL)
    p.add_argument("--max-pages", type=int, default=2, help="最多翻页数；0 表示直到没有下一页")
    p.add_argument("--delay", type=float, default=2.0)
    p.add_argument("--timeout", type=float, default=12.0)
    p.set_defaults(func=collect_list_formats)

    p = sub.add_parser("collect-detail-fields", help="连接已登录 Chrome，断点采集详情页字段")
    add_common_args(p)
    p.add_argument("--debugger-address", default="127.0.0.1:9222")
    p.add_argument("--limit", type=int, default=0, help="最多采集多少条；0 表示不限制")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--delay", type=float, default=2.0)
    p.add_argument("--timeout", type=float, default=12.0)
    p.add_argument("--min-text-len", type=int, default=80)
    p.add_argument("--sync-every", type=int, default=10)
    p.add_argument("--force", action="store_true", help="忽略 checkpoint，重新采集")
    p.add_argument("--retry-failed", action="store_true", help="跳过成功项，重试失败项")
    p.add_argument("--stop-on-block", action="store_true", help="遇到空白/疑似拦截页面时停止")
    add_detail_collection_tuning_args(p)
    p.set_defaults(func=collect_detail_fields)

    p = sub.add_parser("collect-web-fields", help="统一采集列表格式和详情页最低字段")
    add_common_args(p)
    p.add_argument("--debugger-address", default="127.0.0.1:9222")
    p.add_argument("--list-url", default=LIST_URL)
    p.add_argument("--list-pages", type=int, default=0, help="先扫描多少页列表格式；0 表示跳过列表阶段")
    p.add_argument("--limit", type=int, default=0, help="最多采集多少条详情页；0 表示不限制")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--delay", type=float, default=3.0)
    p.add_argument("--timeout", type=float, default=12.0)
    p.add_argument("--min-text-len", type=int, default=80)
    p.add_argument("--sync-every", type=int, default=20)
    p.add_argument("--force", action="store_true", help="忽略 checkpoint，重新采集")
    p.add_argument("--retry-failed", action="store_true", help="跳过成功项，重试失败项")
    p.add_argument("--stop-on-block", action="store_true", help="遇到空白/疑似拦截页面时停止")
    add_detail_collection_tuning_args(p)
    p.set_defaults(func=collect_web_fields)

    p = sub.add_parser("run-web-batches", help="按批次持续采集网页字段，并在每批后做质量检查")
    add_common_args(p)
    p.add_argument("--debugger-address", default="127.0.0.1:9222")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--max-batches", type=int, default=0, help="最多跑多少批；0 表示直到完成或出错")
    p.add_argument("--delay", type=float, default=3.0)
    p.add_argument("--timeout", type=float, default=12.0)
    p.add_argument("--min-text-len", type=int, default=80)
    p.add_argument("--sync-every", type=int, default=20)
    p.add_argument("--force", action="store_true", help="忽略 checkpoint，重新采集")
    p.add_argument("--retry-failed", action="store_true", help="跳过成功项，重试失败项")
    p.add_argument("--stop-on-block", action="store_true", help="遇到空白/疑似拦截页面时停止")
    add_detail_collection_tuning_args(p)
    p.add_argument("--sleep-between-batches", type=float, default=10.0)
    p.set_defaults(func=run_web_batches)

    p = sub.add_parser("refresh-outputs", help="用当前断点表重新生成增强主表和最低字段表")
    add_common_args(p)
    p.set_defaults(func=refresh_outputs)

    p = sub.add_parser("inspect-minimum", help="检查最近一批最低字段表质量")
    add_common_args(p)
    p.add_argument("--last", type=int, default=500, help="检查最近成功采集的多少条")
    p.set_defaults(func=inspect_minimum)

    p = sub.add_parser("fit-model", help="训练 sklearn 潜力模型并计算沉睡度")
    add_common_args(p)
    p.add_argument("--limit", type=int, default=0, help="模型 smoke test 行数；0 表示全量")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--max-text-features", type=int, default=5000)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--candidate-potential-threshold", type=float, default=0.70)
    p.add_argument("--candidate-actual-threshold", type=float, default=0.50)
    p.set_defaults(func=fit_model)

    p = sub.add_parser("export-workbook", help="汇总模型结果工作簿")
    add_common_args(p)
    p.set_defaults(func=export_workbook)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except Exception as exc:
        print(f"[error] {repr(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
