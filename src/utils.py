import csv
import os
import random
from datetime import datetime

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_result(csv_path: str, row: dict) -> None:
    """Append one experiment-result row to a CSV, creating it (with header) if needed.

    New keys added in later runs are handled by rewriting the header union.
    """
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), **row}
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    existing_rows: list[dict] = []
    fieldnames: list[str] = list(row.keys())
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            old_fields = reader.fieldnames or []
        fieldnames = list(old_fields) + [k for k in row if k not in old_fields]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for r in existing_rows:
            writer.writerow(r)
        writer.writerow(row)
