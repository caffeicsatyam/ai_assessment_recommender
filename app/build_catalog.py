import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from collections import Counter

try:
    from app.models import CatalogRecord
    from app.classify import classify
except ImportError:
    from .models import CatalogRecord
    from .classify import classify


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 build_catalog.py <input.json> <output.json>")
        raise SystemExit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = []
    n_parse_errors = 0
    for r in raw:
        try:
            records.append(classify(CatalogRecord(**r)))
        except Exception as e:
            n_parse_errors += 1
            print(
                f"  [skip] failed to parse record: {r.get('name', '?')} -> {e}",
                file=sys.stderr,
            )
    counts = Counter()
    for rec in records:
        if rec.is_job_solution:
            counts["job_solution"] += 1
        elif rec.is_report_only:
            counts["report_only"] += 1
        else:
            counts["individual_test"] += 1
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in records], f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
