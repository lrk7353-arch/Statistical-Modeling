# -*- coding: utf-8 -*-
"""Sidecar monitor for the Shanghai data collection run.

This script is read-only with respect to collection data. It watches the
checkpoint/log files and writes batch reports without changing crawler behavior.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


CORE_COLUMNS = ["download_formats", "record_count", "data_size", "field_names"]
TEXT_COLUMNS = ["field_names", "data_size", "detail_spatial_scope", "detail_time_scope", "备注", "scrape_error"]


def normalize(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def is_missing(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().eq("")


def has_mojibake(value: object) -> bool:
    text = normalize(value)
    if not text:
        return False
    return bool(re.search(r"锟|Ã|Â|�|閺|闁|氓|茂驴陆", text))


def read_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    last_error: Exception | None = None
    for _ in range(6):
        try:
            df = pd.read_csv(path, dtype={"数据集ID": str}, encoding="utf-8-sig")
            break
        except (pd.errors.ParserError, OSError, UnicodeDecodeError) as exc:
            # The collector rewrites the checkpoint frequently. If the monitor
            # reads during a write, pandas can see a partial CSV; wait and retry.
            last_error = exc
            time.sleep(2)
    else:
        raise last_error if last_error is not None else RuntimeError("checkpoint read failed")
    if "scraped_at" in df.columns:
        df["scraped_at"] = pd.to_datetime(df["scraped_at"], errors="coerce")
    return df


def read_log_text(path: Path, max_chars: int = 50000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[-max_chars:]


def latest_progress(stderr_text: str) -> str:
    lines = [line for line in stderr_text.splitlines() if "detail pages:" in line]
    return lines[-1] if lines else ""


def collect_window(df: pd.DataFrame, start_success: int, end_success: int) -> pd.DataFrame:
    ok = df[df.get("scrape_status", "").astype(str).eq("success")].copy()
    if ok.empty:
        return ok
    if "scraped_at" in ok.columns:
        ok = ok.sort_values("scraped_at")
    return ok.iloc[start_success:end_success].copy()


def summarize_window(window: pd.DataFrame, failed: pd.DataFrame, start_success: int, end_success: int) -> dict:
    summary: dict[str, object] = {
        "range": f"{start_success}-{end_success}",
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "success_rows": int(len(window)),
        "failed_rows_seen": int(len(failed)),
    }
    if len(window) and "scraped_at" in window.columns:
        times = window["scraped_at"].dropna()
        if len(times) >= 2:
            seconds = (times.max() - times.min()).total_seconds()
            summary["elapsed_seconds"] = round(seconds, 1)
            summary["elapsed_minutes"] = round(seconds / 60, 2)
            summary["avg_seconds_per_row"] = round(seconds / max(1, len(window) - 1), 2)
        summary["first_scraped_at"] = str(times.min()) if len(times) else ""
        summary["last_scraped_at"] = str(times.max()) if len(times) else ""
    for col in CORE_COLUMNS:
        if col in window.columns:
            summary[f"missing_{col}"] = int(is_missing(window[col]).sum())
        else:
            summary[f"missing_{col}"] = int(len(window))
    if "field_count" in window.columns:
        summary["zero_field_count"] = int((pd.to_numeric(window["field_count"], errors="coerce").fillna(0) <= 0).sum())
    text_cols = [col for col in TEXT_COLUMNS if col in window.columns]
    if text_cols and len(window):
        summary["mojibake_rows"] = int(window[text_cols].apply(lambda row: any(has_mojibake(v) for v in row), axis=1).sum())
    else:
        summary["mojibake_rows"] = 0
    missing_any = pd.Series(False, index=window.index)
    for col in CORE_COLUMNS:
        if col in window.columns:
            missing_any = missing_any | is_missing(window[col])
    summary["core_complete_rows"] = int((~missing_any).sum()) if len(window) else 0
    summary["core_complete_rate"] = round(summary["core_complete_rows"] / len(window), 4) if len(window) else 0
    return summary


def write_report(report_dir: Path, summary: dict, window: pd.DataFrame, stdout_text: str, stderr_text: str) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{summary['range']}.txt"
    missing_rows = pd.Series(False, index=window.index)
    for col in CORE_COLUMNS:
        if col in window.columns:
            missing_rows = missing_rows | is_missing(window[col])
    sample_cols = [col for col in ["数据集ID", "detail_url", *CORE_COLUMNS, "备注", "scrape_error", "scraped_at"] if col in window.columns]
    missing_text = ""
    if len(window) and missing_rows.any():
        missing_text = window.loc[missing_rows, sample_cols].head(20).to_string(index=False)
    content = [
        "上海公共数据网页字段采集批次报告",
        "",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "",
        "最近 stdout 摘要:",
        stdout_text[-3000:],
        "",
        "最近 stderr 进度:",
        latest_progress(stderr_text),
        "",
        "核心字段缺失样例:",
        missing_text or "无",
        "",
    ]
    report_path.write_text("\n".join(content), encoding="utf-8")
    return report_path


def maybe_write_alert(alert_path: Path, checkpoint: pd.DataFrame, stdout_text: str, stderr_text: str, last_alert_key: str) -> str:
    ok = checkpoint[checkpoint.get("scrape_status", "").astype(str).eq("success")].copy()
    if ok.empty:
        return last_alert_key
    if "scraped_at" in ok.columns:
        ok = ok.sort_values("scraped_at")
    recent = ok.tail(50).copy()
    missing_counts = {}
    for col in CORE_COLUMNS:
        missing_counts[col] = int(is_missing(recent[col]).sum()) if col in recent.columns else len(recent)
    failed_count = int((checkpoint.get("scrape_status", "").astype(str) == "failed").sum()) if "scrape_status" in checkpoint.columns else 0
    mojibake = 0
    text_cols = [col for col in TEXT_COLUMNS if col in recent.columns]
    if text_cols and len(recent):
        mojibake = int(recent[text_cols].apply(lambda row: any(has_mojibake(v) for v in row), axis=1).sum())
    restart_seen = "[restart]" in stdout_text[-5000:]
    block_seen = "possible anti-crawl block" in stdout_text[-5000:] or "empty or blocked" in stdout_text[-5000:]
    issues = []
    for col, count in missing_counts.items():
        if count >= 3:
            issues.append(f"最近50条 {col} 缺失 {count} 条")
    if mojibake:
        issues.append(f"最近50条疑似乱码 {mojibake} 条")
    if restart_seen:
        issues.append("最近日志出现自动重启")
    if block_seen:
        issues.append("最近日志疑似出现拦截/空白页")
    if not issues:
        return last_alert_key
    last_time = normalize(recent.get("scraped_at", pd.Series([""])).iloc[-1]) if len(recent) else ""
    key = f"{last_time}|{'|'.join(issues)}"
    if key == last_alert_key:
        return last_alert_key
    alert_path.parent.mkdir(parents=True, exist_ok=True)
    with alert_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] " + "；".join(issues) + "\n")
        fh.write("stderr进度: " + latest_progress(stderr_text) + "\n")
    return key


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("--stderr-log", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--start-success", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--poll-seconds", type=float, default=120)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    stdout_path = Path(args.stdout_log)
    stderr_path = Path(args.stderr_log)
    report_dir = Path(args.report_dir)
    alert_path = report_dir / "web_collection_alerts.txt"

    next_start = args.start_success
    next_end = next_start + args.batch_size
    last_alert_key = ""
    while True:
        checkpoint = read_checkpoint(checkpoint_path)
        stdout_text = read_log_text(stdout_path)
        stderr_text = read_log_text(stderr_path)
        if not checkpoint.empty:
            last_alert_key = maybe_write_alert(alert_path, checkpoint, stdout_text, stderr_text, last_alert_key)
            ok_count = int((checkpoint.get("scrape_status", "").astype(str) == "success").sum())
            while ok_count >= next_end:
                failed = checkpoint[checkpoint.get("scrape_status", "").astype(str).eq("failed")].copy()
                window = collect_window(checkpoint, next_start, next_end)
                summary = summarize_window(window, failed, next_start, next_end)
                report_path = write_report(report_dir, summary, window, stdout_text, stderr_text)
                print(f"[monitor] report written: {report_path}", flush=True)
                next_start = next_end
                next_end = next_start + args.batch_size
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
