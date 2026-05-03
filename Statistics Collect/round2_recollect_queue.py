from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import pandas as pd


def load_pipeline():
    script = Path(__file__).with_name("sh_data_pipeline.py")
    spec = importlib.util.spec_from_file_location("sh_data_pipeline_round2", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sh_data_pipeline_round2"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def emptyish(value: object) -> bool:
    if pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def missing_core_fields(row: pd.Series | dict, core: list[str]) -> list[str]:
    return [field for field in core if emptyish(row.get(field, ""))]


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8-sig")
    rows.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description="Recollect selected Shanghai data detail pages from a queue.")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--debugger-address", default="127.0.0.1:9222")
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--min-text-len", type=int, default=80)
    parser.add_argument("--detail-max-attempts", type=int, default=3)
    parser.add_argument("--navigate-wait-seconds", type=float, default=8.0)
    parser.add_argument("--sync-every", type=int, default=20)
    parser.add_argument("--result-flush-every", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    pipe = load_pipeline()
    paths = pipe.make_paths(argparse.Namespace(input=pipe.DEFAULT_CATALOG, output_dir=pipe.DEFAULT_OUTPUT_DIR))
    queue_path = Path(args.queue)
    queue = pd.read_csv(queue_path, dtype=str, encoding="utf-8-sig")
    id_col = next(col for col in queue.columns if col.endswith("ID"))
    if args.limit:
        queue = queue.head(args.limit).copy()

    checkpoint = pipe.read_checkpoint(paths.checkpoint_csv)
    cp_id_col = next(col for col in checkpoint.columns if col.endswith("ID"))
    checkpoint = checkpoint.drop_duplicates(subset=[cp_id_col], keep="last")
    old_by_id = checkpoint.set_index(cp_id_col, drop=False)
    core = ["download_formats", "record_count", "data_size", "field_names"]

    result_path = paths.output_dir / "round2_recollect_results.csv"
    if result_path.exists():
        result_path.unlink()

    driver = pipe.connect_chrome(args.debugger_address, navigate_wait_seconds=args.navigate_wait_seconds)
    result_buffer: list[dict] = []
    saved_since_sync = 0
    processed = 0
    improved = 0
    recovered_unavailable = 0
    still_unavailable = 0
    skipped_regression = 0
    failed = 0

    try:
        for idx, row in queue.iterrows():
            dataset_id = pipe.normalize_id(row[id_col])
            old = old_by_id.loc[dataset_id] if dataset_id in old_by_id.index else pd.Series(dtype=object)
            if isinstance(old, pd.DataFrame):
                old = old.iloc[-1]
            old_status = str(old.get("scrape_status", ""))
            old_missing = missing_core_fields(old, core)
            url = pipe.normalize_space(row.get("detail_url", "")) or pipe.normalize_space(old.get("detail_url", ""))
            type_code = pipe.normalize_space(row.get("type_code", "")) or pipe.normalize_space(old.get("type_code", ""))
            if not url:
                url = pipe.build_detail_url(dataset_id, type_code)

            started = time.time()
            action = "noop"
            new_status = ""
            new_missing: list[str] = []
            error = ""
            try:
                record = pipe.parse_detail_with_retries(driver, url, dataset_id, args)
                new_status = str(record.get("scrape_status", ""))
                new_missing = pipe.missing_core_detail_fields(record)
                if old_status == "success" and new_status != "success":
                    action = "skip_success_regression"
                    skipped_regression += 1
                else:
                    pipe.upsert_checkpoint(paths.checkpoint_csv, [record])
                    saved_since_sync += 1
                    if old_status == "unavailable" and new_status == "success":
                        recovered_unavailable += 1
                    elif new_status == "unavailable":
                        still_unavailable += 1
                    elif set(new_missing) < set(old_missing):
                        improved += 1
                    action = "upsert"
            except Exception as exc:
                failed += 1
                error = repr(exc)
                action = "failed"

            processed += 1
            duration = time.time() - started
            result_buffer.append(
                {
                    "row_no": int(idx) + 1,
                    "dataset_id": dataset_id,
                    "retry_reason": row.get("retry_reason", ""),
                    "old_status": old_status,
                    "old_missing_core": ",".join(old_missing),
                    "new_status": new_status,
                    "new_missing_core": ",".join(new_missing),
                    "action": action,
                    "duration_seconds": round(duration, 2),
                    "error": error,
                }
            )
            if len(result_buffer) >= max(1, args.result_flush_every):
                append_csv(result_path, result_buffer)
            if saved_since_sync >= args.sync_every:
                pipe.sync_master_from_checkpoint(paths)
                saved_since_sync = 0
            print(
                f"[round2] {processed}/{len(queue)} id={dataset_id} "
                f"old={old_status} new={new_status} old_missing={','.join(old_missing)} "
                f"new_missing={','.join(new_missing)} action={action} dt={duration:.1f}s",
                flush=True,
            )
            if args.delay:
                time.sleep(args.delay)
    finally:
        append_csv(result_path, result_buffer)
        pipe.sync_master_from_checkpoint(paths)

    print(
        f"[round2-done] processed={processed} improved={improved} recovered_unavailable={recovered_unavailable} "
        f"still_unavailable={still_unavailable} skipped_regression={skipped_regression} failed={failed}",
        flush=True,
    )
    print(f"[round2-done] results={result_path}", flush=True)


if __name__ == "__main__":
    main()
