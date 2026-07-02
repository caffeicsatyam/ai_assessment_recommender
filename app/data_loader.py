import json
import re
from pathlib import Path
from typing import Optional

try:
    from .models import CatalogRecord
    from .retrieval import CatalogIndex
except ImportError:
    from app.models import CatalogRecord
    from app.retrieval import CatalogIndex

_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = _ROOT / "data" / "catalog_clean.json"
INDEX_PATH = _ROOT / "data" / "index_data"

_catalog_by_id: dict[str, CatalogRecord] = {}
_index: Optional[CatalogIndex] = None


def get_catalog() -> dict[str, CatalogRecord]:
    global _catalog_by_id
    if not _catalog_by_id:
        if CATALOG_PATH.exists():
            with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            _catalog_by_id = {r["entity_id"]: CatalogRecord(**r) for r in raw}
        else:
            raw_path = _ROOT / "data" / "shl_product_catalog.json"
            if raw_path.exists():
                with open(raw_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                from .classify import classify

                _catalog_by_id = {}
                for r in raw:
                    try:
                        rec = classify(CatalogRecord(**r))
                        _catalog_by_id[rec.entity_id] = rec
                    except Exception:
                        pass
    return _catalog_by_id


def get_catalog_by_name() -> dict[str, CatalogRecord]:
    catalog = get_catalog()
    return {rec.name.lower().strip(): rec for rec in catalog.values()}


def get_index() -> CatalogIndex:
    global _index
    if _index is None:
        _index = CatalogIndex.load(str(INDEX_PATH))
    return _index


def parse_duration(duration_str: str) -> int:
    if not duration_str:
        return 0
    match = re.search(r"\d+", duration_str)
    if match:
        return int(match.group(0))
    return 0
