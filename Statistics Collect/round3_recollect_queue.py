from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


CORE4 = ["download_formats", "record_count", "data_size", "field_names"]
CORE6 = [*CORE4, "detail_spatial_scope", "detail_time_scope"]
FORMAT_FLAGS = ["has_rdf", "has_xml", "has_csv", "has_json", "has_xlsx", "format_count"]
FIELD_RELATED = [
    "field_names",
    "field_count",
    "has_time_field",
    "has_geo_field",
    "field_description_count",
    "has_standard_field_description",
    "has_data_sample",
    "sample_field_headers",
]
OTHER_DETAIL_FIELDS = [
    "api_need_apply",
    "recommended_dataset_count",
    "recommended_dataset_names",
    "rating_score",
    "comment_count",
]
LATEST_FIELDS = {"scrape_status", "scrape_error", "scraped_at", "备注", "澶囨敞"}


def load_pipeline():
    script = Path(__file__).with_name("sh_data_pipeline.py")
    spec = importlib.util.spec_from_file_location("sh_data_pipeline_round3", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sh_data_pipeline_round3"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def emptyish(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def text_value(value: object) -> str:
    return "" if emptyish(value) else str(value).strip()


def split_terms(value: object) -> list[str]:
    text = text_value(value)
    if not text:
        return []
    return [part.strip() for part in re.split(r"[;\uff1b\n\r]+", text) if part.strip()]


def number_value(value: object) -> float | None:
    if emptyish(value):
        return None
    try:
        number = float(str(value).strip())
    except Exception:
        return None
    if math.isnan(number):
        return None
    return number


def field_count(row: pd.Series | dict) -> int:
    explicit = number_value(row.get("field_count", ""))
    if explicit is not None and explicit >= 0:
        return int(explicit)
    explicit = number_value(row.get("field_count_detected", ""))
    if explicit is not None and explicit >= 0:
        return int(explicit)
    return len(split_terms(row.get("field_names", "")))


def format_count(row: pd.Series | dict) -> int:
    explicit = number_value(row.get("format_count", ""))
    if explicit is not None and explicit >= 0:
        return int(explicit)
    return len(split_terms(row.get("download_formats", "")))


def missing_fields(row: pd.Series | dict, fields: list[str]) -> list[str]:
    return [field for field in fields if emptyish(row.get(field, ""))]


def id_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if str(col).endswith("ID"):
            return col
    raise ValueError("Could not find an ID column.")


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8-sig")
    rows.clear()


def has_mojibake(pipe, row: pd.Series | dict) -> bool:
    for field in ["field_names", "data_size", "detail_spatial_scope", "detail_time_scope", "scrape_error", "备注", "澶囨敞"]:
        if field in row and pipe.has_mojibake(row.get(field, "")):
            return True
    return False


def has_suspicious_fields(pipe, row: pd.Series | dict) -> bool:
    return bool(text_value(row.get("field_names", "")) and pipe.has_suspicious_field_names(row.get("field_names", "")))


def quality_issues(pipe, record: pd.Series | dict, reason: str) -> list[str]:
    issues: list[str] = []
    status = text_value(record.get("scrape_status", ""))
    if status != "success":
        issues.append(status or "non_success")
        return issues
    missing = missing_fields(record, CORE6)
    issues.extend(f"missing_{field}" for field in missing)
    if has_mojibake(pipe, record):
        issues.append("mojibake")
    if has_suspicious_fields(pipe, record):
        issues.append("suspicious_field_names")
    if "field_terms_too_few" in str(reason) and field_count(record) <= 2:
        issues.append("field_terms_too_few")
    return issues


def quality_score(pipe, record: pd.Series | dict, reason: str) -> int:
    status = text_value(record.get("scrape_status", ""))
    score = 0
    if status == "success":
        score += 10000
    elif status == "unavailable":
        score -= 1000
    else:
        score -= 2000
    score -= 600 * len(missing_fields(record, CORE6))
    if has_mojibake(pipe, record):
        score -= 800
    if has_suspicious_fields(pipe, record):
        score -= 300
    if "field_terms_too_few" in str(reason) and field_count(record) <= 2:
        score -= 250
    score += min(200, field_count(record) * 8)
    score += min(50, format_count(record) * 5)
    return score


def better_field_names(pipe, old: pd.Series | dict, new: pd.Series | dict) -> bool:
    new_names = text_value(new.get("field_names", ""))
    if not new_names:
        return False
    old_names = text_value(old.get("field_names", ""))
    if not old_names:
        return True
    old_suspicious = has_suspicious_fields(pipe, old)
    new_suspicious = has_suspicious_fields(pipe, new)
    if old_suspicious and not new_suspicious:
        return True
    return field_count(new) > field_count(old)


def better_formats(old: pd.Series | dict, new: pd.Series | dict) -> bool:
    new_formats = text_value(new.get("download_formats", ""))
    if not new_formats:
        return False
    old_formats = text_value(old.get("download_formats", ""))
    if not old_formats:
        return True
    return format_count(new) > format_count(old)


def build_selective_success_update(
    pipe, dataset_id: str, id_col: str, old: pd.Series | dict, best: dict
) -> tuple[dict | None, list[str]]:
    update = {
        id_col: dataset_id,
        "detail_url": best.get("detail_url", old.get("detail_url", "")),
        "scrape_status": "success",
        "scrape_error": "",
        "scraped_at": best.get("scraped_at", datetime.now().isoformat(timespec="seconds")),
    }
    if "备注" in best:
        update["备注"] = best.get("备注", "")
    if "澶囨敞" in best:
        update["澶囨敞"] = best.get("澶囨敞", "")

    changed: list[str] = []
    if better_formats(old, best):
        for field in ["download_formats", *FORMAT_FLAGS]:
            if field in best:
                update[field] = best.get(field)
        changed.append("download_formats")

    if better_field_names(pipe, old, best):
        for field in FIELD_RELATED:
            if field in best:
                update[field] = best.get(field)
        changed.append("field_names")

    for field in ["record_count", "data_size", "detail_spatial_scope", "detail_time_scope"]:
        if emptyish(old.get(field, "")) and not emptyish(best.get(field, "")):
            update[field] = best.get(field)
            changed.append(field)

    for field in OTHER_DETAIL_FIELDS:
        if field in best and emptyish(old.get(field, "")) and not emptyish(best.get(field, "")):
            update[field] = best.get(field)

    if not changed:
        return None, changed
    return update, changed


def full_success_update(dataset_id: str, id_col: str, best: dict) -> dict:
    record = dict(best)
    record[id_col] = dataset_id
    return record


def connect_or_start_chrome(pipe, args: argparse.Namespace):
    try:
        return pipe.connect_chrome(args.debugger_address, navigate_wait_seconds=args.navigate_wait_seconds)
    except Exception:
        print("[round3] Chrome not connected; starting controlled Chrome.", flush=True)
        return pipe.restart_controlled_chrome(None, args)


def maybe_restart(pipe, driver, args: argparse.Namespace, durations: list[float], last_restart_at: float, restart_count: int, remaining: bool):
    window = max(1, int(args.restart_window))
    if len(durations) > window:
        del durations[:-window]
    if not (args.auto_restart_on_slow and remaining and len(durations) >= window):
        return driver, last_restart_at, restart_count
    avg_duration = sum(durations[-window:]) / window
    allowed_interval = time.time() - last_restart_at >= args.min_restart_interval
    allowed_count = args.max_restarts <= 0 or restart_count < args.max_restarts
    if avg_duration > args.slow_threshold_seconds and allowed_interval and allowed_count:
        print(
            f"[restart] avg_last_{window}={avg_duration:.1f}s > {args.slow_threshold_seconds:.1f}s; restarting Chrome",
            flush=True,
        )
        driver = pipe.restart_controlled_chrome(driver, args)
        durations.clear()
        return driver, time.time(), restart_count + 1
    return driver, last_restart_at, restart_count


def write_final_summary(pipe, paths, result_path: Path, summary_path: Path, residual_path: Path) -> None:
    pipe.sync_master_from_checkpoint(paths)
    master = pipe.read_master(paths.master_xlsx)
    checkpoint = pipe.read_checkpoint(paths.checkpoint_csv)
    master_id_col = id_column(master)
    cp_id_col = id_column(checkpoint)
    master = master.copy()
    master["__master_row_number"] = range(1, len(master) + 1)
    cp = checkpoint.drop_duplicates(subset=[cp_id_col], keep="last").copy()
    joined = master[[master_id_col, "__master_row_number"]].merge(
        cp, left_on=master_id_col, right_on=cp_id_col, how="left", suffixes=("", "__cp")
    )
    if cp_id_col != master_id_col and cp_id_col in joined.columns:
        joined = joined.drop(columns=[cp_id_col])
    status = joined.get("scrape_status", pd.Series([""] * len(joined))).fillna("").astype(str)
    issues_by_row: list[str] = []
    for _, row in joined.iterrows():
        issues = []
        if row.get("scrape_status", "") == "unavailable":
            issues.append("unavailable")
        elif row.get("scrape_status", "") != "success":
            issues.append("non_success")
        issues.extend(f"missing_{field}" for field in missing_fields(row, CORE6))
        if has_mojibake(pipe, row):
            issues.append("mojibake")
        if has_suspicious_fields(pipe, row):
            issues.append("suspicious_field_names")
        if row.get("scrape_status", "") == "success" and field_count(row) <= 2:
            issues.append("field_terms_too_few")
        issues_by_row.append(";".join(dict.fromkeys(issues)))
    joined["round3_residual_issues"] = issues_by_row
    residual = joined[joined["round3_residual_issues"].astype(str).str.strip().ne("")].copy()
    keep_cols = [
        "__master_row_number",
        master_id_col,
        "detail_url",
        "scrape_status",
        "download_formats",
        "record_count",
        "data_size",
        "field_names",
        "field_count",
        "detail_spatial_scope",
        "detail_time_scope",
        "scrape_error",
        "scraped_at",
        "round3_residual_issues",
    ]
    keep_cols = [col for col in keep_cols if col in residual.columns]
    residual[keep_cols].to_csv(residual_path, index=False, encoding="utf-8-sig")

    results = pd.read_csv(result_path, dtype=str, encoding="utf-8-sig") if result_path.exists() else pd.DataFrame()
    action_counts = results["action"].value_counts(dropna=False).to_dict() if "action" in results.columns else {}
    final_status_counts = status.value_counts(dropna=False).to_dict()
    issue_counter = Counter()
    for issue_text in residual["round3_residual_issues"].fillna("").astype(str):
        for issue in [x for x in issue_text.split(";") if x]:
            issue_counter[issue] += 1

    lines = [
        "Round 3 recollect final summary",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"master_rows={len(master)}",
        f"checkpoint_rows={len(checkpoint)}",
        f"result_rows={len(results)}",
        f"final_status_counts={final_status_counts}",
        f"action_counts={action_counts}",
        f"residual_rows={len(residual)}",
        f"residual_issue_counts={dict(issue_counter)}",
        f"results={result_path}",
        f"residual={residual_path}",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality-first final recollect for Shanghai data rows.")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--debugger-address", default="127.0.0.1:9222")
    parser.add_argument("--delay", type=float, default=0.6)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--min-text-len", type=int, default=80)
    parser.add_argument("--detail-max-attempts", type=int, default=1)
    parser.add_argument("--quality-checks", type=int, default=3)
    parser.add_argument("--navigate-wait-seconds", type=float, default=6.0)
    parser.add_argument("--sync-every", type=int, default=20)
    parser.add_argument("--result-flush-every", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--auto-restart-on-slow", action="store_true")
    parser.add_argument("--restart-window", type=int, default=20)
    parser.add_argument("--slow-threshold-seconds", type=float, default=20.0)
    parser.add_argument("--min-restart-interval", type=float, default=240.0)
    parser.add_argument("--max-restarts", type=int, default=0)
    parser.add_argument("--restart-wait", type=float, default=4.0)
    parser.add_argument("--chrome-path", default="")
    parser.add_argument("--chrome-user-data-dir", default="")
    parser.add_argument("--list-url", default="")
    args = parser.parse_args()

    pipe = load_pipeline()
    paths = pipe.make_paths(argparse.Namespace(input=pipe.DEFAULT_CATALOG, output_dir=pipe.DEFAULT_OUTPUT_DIR))
    queue_path = Path(args.queue)
    queue = pd.read_csv(queue_path, dtype=str, encoding="utf-8-sig")
    q_id_col = id_column(queue)
    if args.limit:
        queue = queue.head(args.limit).copy()

    checkpoint = pipe.read_checkpoint(paths.checkpoint_csv)
    cp_id_col = id_column(checkpoint)
    checkpoint = checkpoint.drop_duplicates(subset=[cp_id_col], keep="last")
    old_by_id = checkpoint.set_index(cp_id_col, drop=False)
    master = pipe.read_master(paths.master_xlsx)
    master_id_col = id_column(master)
    row_no_by_id = {pipe.normalize_id(v): i + 1 for i, v in enumerate(master[master_id_col].tolist())}

    result_path = paths.output_dir / "round3_recollect_results.csv"
    summary_path = paths.output_dir / "collection_reports" / "round3_recollect_final_summary.txt"
    residual_path = paths.output_dir / "round3_residual_missing_or_unavailable.csv"
    for path in [result_path, summary_path, residual_path]:
        if path.exists():
            path.unlink()

    driver = connect_or_start_chrome(pipe, args)
    result_buffer: list[dict] = []
    durations: list[float] = []
    saved_since_sync = 0
    last_restart_at = time.time()
    restart_count = 0
    counters = Counter()

    try:
        for idx, row in queue.iterrows():
            row_started = time.time()
            dataset_id = pipe.normalize_id(row[q_id_col])
            reason = text_value(row.get("third_retry_reason", row.get("retry_reason", "")))
            old = old_by_id.loc[dataset_id] if dataset_id in old_by_id.index else pd.Series(dtype=object)
            if isinstance(old, pd.DataFrame):
                old = old.iloc[-1]
            old_status = text_value(old.get("scrape_status", ""))
            old_issues = quality_issues(pipe, old, reason) if len(old) else ["not_in_checkpoint"]
            url = text_value(row.get("detail_url", "")) or text_value(old.get("detail_url", ""))
            type_code = text_value(row.get("type_code", "")) or text_value(old.get("type_code", ""))
            if not url:
                url = pipe.build_detail_url(dataset_id, type_code)

            best_record: dict | None = None
            best_issues: list[str] = []
            attempt_errors: list[str] = []
            per_attempts: list[str] = []
            for check_no in range(1, max(1, args.quality_checks) + 1):
                attempt_args = copy.copy(args)
                attempt_args.detail_max_attempts = max(1, args.detail_max_attempts)
                try:
                    record = pipe.parse_detail_with_retries(driver, url, dataset_id, attempt_args)
                    record["detail_url"] = url
                    issues = quality_issues(pipe, record, reason)
                    per_attempts.append(f"{check_no}:{record.get('scrape_status','')}:{'|'.join(issues) or 'ok'}:fc={field_count(record)}")
                    if best_record is None or quality_score(pipe, record, reason) > quality_score(pipe, best_record, reason):
                        best_record = dict(record)
                        best_issues = issues
                    if not issues:
                        break
                    time.sleep(0.6 + 0.3 * check_no)
                except Exception as exc:
                    attempt_errors.append(f"{check_no}:{repr(exc)}")
                    per_attempts.append(f"{check_no}:error:{repr(exc)}")
                    time.sleep(0.8 + 0.4 * check_no)

            action = "failed"
            changed_fields: list[str] = []
            new_status = ""
            new_issues = ["no_record"]
            if best_record is None:
                counters["failed"] += 1
            else:
                new_status = text_value(best_record.get("scrape_status", ""))
                new_issues = best_issues
                if old_status == "success" and new_status != "success":
                    action = "skip_success_regression"
                    counters[action] += 1
                elif old_status != "success" and new_status == "success":
                    pipe.upsert_checkpoint(paths.checkpoint_csv, [full_success_update(dataset_id, cp_id_col, best_record)])
                    saved_since_sync += 1
                    action = "upsert_recovered_success"
                    changed_fields = ["full_success"]
                    counters[action] += 1
                elif old_status == "success" and new_status == "success":
                    update, changed_fields = build_selective_success_update(pipe, dataset_id, cp_id_col, old, best_record)
                    if update is not None:
                        pipe.upsert_checkpoint(paths.checkpoint_csv, [update])
                        saved_since_sync += 1
                        action = "upsert_improved_success"
                    else:
                        action = "checked_no_improvement"
                    counters[action] += 1
                else:
                    pipe.upsert_checkpoint(paths.checkpoint_csv, [full_success_update(dataset_id, cp_id_col, best_record)])
                    saved_since_sync += 1
                    action = f"upsert_{new_status or 'non_success'}"
                    counters[action] += 1

            if saved_since_sync >= args.sync_every:
                pipe.sync_master_from_checkpoint(paths)
                saved_since_sync = 0

            duration = time.time() - row_started
            durations.append(duration)
            result_buffer.append(
                {
                    "queue_row_no": int(idx) + 1,
                    "master_row_no": row_no_by_id.get(dataset_id, ""),
                    "dataset_id": dataset_id,
                    "third_retry_reason": reason,
                    "old_status": old_status,
                    "old_issues": ";".join(old_issues),
                    "new_status": new_status,
                    "new_issues": ";".join(new_issues),
                    "action": action,
                    "changed_fields": ";".join(changed_fields),
                    "quality_checks": len(per_attempts),
                    "attempts": " || ".join(per_attempts),
                    "duration_seconds": round(duration, 2),
                    "error": " || ".join(attempt_errors),
                }
            )
            if len(result_buffer) >= max(1, args.result_flush_every):
                append_csv(result_path, result_buffer)

            print(
                f"[round3] {int(idx)+1}/{len(queue)} master_row={row_no_by_id.get(dataset_id, '')} "
                f"id={dataset_id} old={old_status} new={new_status} issues={','.join(new_issues)} "
                f"action={action} changed={','.join(changed_fields)} checks={len(per_attempts)} dt={duration:.1f}s",
                flush=True,
            )
            remaining = int(idx) + 1 < len(queue)
            driver, last_restart_at, restart_count = maybe_restart(
                pipe, driver, args, durations, last_restart_at, restart_count, remaining
            )
            if args.delay:
                time.sleep(args.delay)
    finally:
        append_csv(result_path, result_buffer)
        pipe.sync_master_from_checkpoint(paths)
        write_final_summary(pipe, paths, result_path, summary_path, residual_path)

    print(f"[round3-done] counters={dict(counters)}", flush=True)
    print(f"[round3-done] results={result_path}", flush=True)
    print(f"[round3-done] summary={summary_path}", flush=True)
    print(f"[round3-done] residual={residual_path}", flush=True)


if __name__ == "__main__":
    main()
