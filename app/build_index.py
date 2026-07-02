"""
Run this ONCE, offline, to build and persist the FAISS index.

    python3 build_index.py catalog_clean.json index_data/

Produces index_data/catalog.index and index_data/catalog_indexed.json.
Commit these two files to your repo (or bake them into your deploy image)
so the running FastAPI service never has to re-embed the catalog -- it
just calls CatalogIndex.load("index_data/") at startup, which only loads
the model (for embedding live queries) and reads the pre-built index.

Re-run this script only when the underlying catalog changes.
"""
import json
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.models import CatalogRecord
    from app.retrieval import CatalogIndex
except ImportError:
    from .models import CatalogRecord
    from .retrieval import CatalogIndex

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 build_index.py <catalog_clean.json> <output_dir>")
        raise SystemExit(1)

    catalog_path, out_dir = sys.argv[1], sys.argv[2]

    with open(catalog_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = [CatalogRecord(**r) for r in raw]

    print(f"Building index from {len(records)} classified records...")
    index = CatalogIndex.build(records)
    print(f"Indexed {len(index.records)} recommendable records "
          f"(excluded {len(records) - len(index.records)} job-solution/report records)")

    index.save(out_dir)
    print(f"Saved index + records to {out_dir}/")
    print("Commit this directory, or bake it into your deploy image.")
    print("At server startup, call CatalogIndex.load(out_dir) -- do NOT call build() again.")


if __name__ == "__main__":
    main()