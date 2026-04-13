#!/usr/bin/env python3
"""Convert evaluation JSON with {columns, data} into CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _normalize_cell(value: Any) -> Any:
    """Serialize nested objects so each row can be written as plain CSV cells."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def convert_json_to_csv(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    columns = payload.get("columns")
    rows = payload.get("data")

    if not isinstance(columns, list) or not isinstance(rows, list):
        raise ValueError("Input JSON must contain list fields 'columns' and 'data'.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            if not isinstance(row, list):
                raise ValueError("Each item in 'data' must be a list.")
            writer.writerow([_normalize_cell(cell) for cell in row])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a JSON result file with 'columns' and 'data' to CSV."
    )
    parser.add_argument("input", type=Path, help="Path to input JSON file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV path (default: same name as input with .csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = args.input
    output_path = args.output or input_path.with_suffix(".csv")

    convert_json_to_csv(input_path, output_path)
    print(f"Wrote CSV: {output_path}")


if __name__ == "__main__":
    main()