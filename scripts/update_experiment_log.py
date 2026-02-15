#!/usr/bin/env python
"""
从 backtest_report_{period}.md 自动提取关键指标并追加到 data/mx/experiment_log.csv。

用法示例:
python scripts/update_experiment_log.py --period 5min --features-used all_features --key-params "n_trials=20; feature_selection=on"
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import shlex
import subprocess
import sys
from typing import Dict, List


CSV_HEADERS = [
    "date",
    "period",
    "features_used",
    "key_params",
    "accuracy",
    "f1_score",
    "win_rate",
    "profit_factor",
    "max_drawdown",
    "sharpe_ratio",
    "notes",
]

TARGET_KEYS = {
    "accuracy": "accuracy",
    "f1_score": "f1_score",
    "win_rate": "win_rate",
    "profit_factor": "profit_factor",
    "max_drawdown": "max_drawdown",
    "sharpe_ratio": "sharpe_ratio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从回测报告抽取指标并更新 experiment_log.csv")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--period", help="单周期，如 1min/5min/15min")
    group.add_argument("--periods", help="批量周期，逗号分隔，如 1min,5min,15min")
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="记录日期，默认今天")
    parser.add_argument("--features-used", default="all_features", help="本次使用的特征描述")
    parser.add_argument("--key-params", default="", help="关键参数描述")
    parser.add_argument("--notes", default="", help="备注")
    parser.add_argument("--report", default="", help="回测报告路径，默认 data/mx/backtest_report_{period}.md")
    parser.add_argument("--log", default=os.path.join("data", "mx", "experiment_log.csv"), help="日志CSV路径")
    parser.add_argument("--allow-empty", action="store_true", help="报告缺失或未解析到指标时也允许写空值")
    parser.add_argument("--run-pipeline", action="store_true", help="写日志前先运行 run_real_data_pipeline.py")
    parser.add_argument("--pipeline-script", default="run_real_data_pipeline.py", help="流水线脚本路径")
    parser.add_argument("--pipeline-extra-args", default="", help="传给流水线的附加参数，如 \"--n-trials 20\"")
    parser.add_argument("--strict", action="store_true", help="严格模式：任一周期失败立即中止并返回非零")
    return parser.parse_args()


def run_pipeline_for_period(period: str, script_path: str, extra_args: str = "") -> int:
    cmd = [sys.executable, script_path, "--period", period]
    if extra_args.strip():
        cmd.extend(shlex.split(extra_args))

    print(f"run_pipeline: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return int(result.returncode)


def parse_markdown_metrics(report_path: str) -> Dict[str, str]:
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"未找到报告文件: {report_path}")

    with open(report_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    metrics: Dict[str, str] = {}
    table_row_pattern = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$")

    for line in lines:
        if not line.strip().startswith("|"):
            continue
        if "------" in line:
            continue

        m = table_row_pattern.match(line.strip())
        if not m:
            continue

        key = m.group(1).strip()
        value = m.group(2).strip()

        mapped_key = TARGET_KEYS.get(key)
        if mapped_key:
            metrics[mapped_key] = normalize_metric_value(value)

    return metrics


def normalize_metric_value(raw: str) -> str:
    s = raw.strip()
    if not s:
        return ""

    # 处理百分号，如 55.23%
    if s.endswith("%"):
        num = safe_float(s[:-1])
        if num is None:
            return ""
        return f"{num / 100.0:.6f}"

    num = safe_float(s)
    if num is None:
        return ""
    return f"{num:.6f}"


def safe_float(s: str):
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None


def ensure_csv(csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()


def read_existing_rows(csv_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_rows(csv_path: str, rows: List[Dict[str, str]]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def upsert_row(csv_path: str, row: Dict[str, str]) -> str:
    rows = read_existing_rows(csv_path)
    updated = False

    for i, r in enumerate(rows):
        if r.get("date") == row["date"] and r.get("period") == row["period"]:
            rows[i] = {**r, **row}
            updated = True
            break

    if not updated:
        rows.append(row)

    write_rows(csv_path, rows)
    return "updated" if updated else "inserted"


def main() -> int:
    args = parse_args()

    if args.periods:
        periods = [p.strip() for p in args.periods.split(",") if p.strip()]
    else:
        periods = [args.period]

    if not periods:
        print("未提供有效周期，请使用 --period 或 --periods。")
        return 3

    log_path = args.log
    ensure_csv(log_path)

    has_failure = False

    for period in periods:
        if args.run_pipeline:
            ret = run_pipeline_for_period(
                period=period,
                script_path=args.pipeline_script,
                extra_args=args.pipeline_extra_args,
            )
            if ret != 0:
                print(f"流水线执行失败: period={period}, return_code={ret}")
                has_failure = True
                if args.strict:
                    return 1
                continue

        report_path = args.report or os.path.join("data", "mx", f"backtest_report_{period}.md")

        try:
            metrics = parse_markdown_metrics(report_path)
        except FileNotFoundError as e:
            if not args.allow_empty:
                print(str(e))
                print("可使用 --allow-empty 允许写入空指标行。")
                has_failure = True
                if args.strict:
                    return 1
                continue
            metrics = {}

        required_any = [
            metrics.get("accuracy", ""),
            metrics.get("f1_score", ""),
            metrics.get("win_rate", ""),
            metrics.get("profit_factor", ""),
            metrics.get("max_drawdown", ""),
            metrics.get("sharpe_ratio", ""),
        ]

        if not any(required_any) and not args.allow_empty:
            print(f"报告已读取但未解析到目标指标: {report_path}")
            print("可使用 --allow-empty 允许写入空指标行。")
            has_failure = True
            if args.strict:
                return 1
            continue

        row = {
            "date": args.date,
            "period": period,
            "features_used": args.features_used,
            "key_params": args.key_params,
            "accuracy": metrics.get("accuracy", ""),
            "f1_score": metrics.get("f1_score", ""),
            "win_rate": metrics.get("win_rate", ""),
            "profit_factor": metrics.get("profit_factor", ""),
            "max_drawdown": metrics.get("max_drawdown", ""),
            "sharpe_ratio": metrics.get("sharpe_ratio", ""),
            "notes": args.notes,
        }

        mode = upsert_row(log_path, row)
        print(f"{mode}: {log_path}")
        print(f"period: {period}")
        print(f"report: {report_path}")
        print(
            "parsed_metrics:",
            {
                "accuracy": row["accuracy"],
                "f1_score": row["f1_score"],
                "win_rate": row["win_rate"],
                "profit_factor": row["profit_factor"],
                "max_drawdown": row["max_drawdown"],
                "sharpe_ratio": row["sharpe_ratio"],
            },
        )

    return 1 if has_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
