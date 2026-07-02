"""
Data loading and caching for the SHL catalog and FAISS index.

All catalog / index access should go through the functions here
so that the data is loaded exactly once and shared across the app.
"""

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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = _ROOT / "data" / "catalog_clean.json"
INDEX_PATH = _ROOT / "data" / "index_data"

# ---------------------------------------------------------------------------
# Cached singletons
# ---------------------------------------------------------------------------
_catalog_by_id: dict[str, CatalogRecord] = {}
_index: Optional[CatalogIndex] = None


def get_catalog() -> dict[str, CatalogRecord]:
    """Return the full catalog keyed by entity_id (cached after first load)."""
    global _catalog_by_id
    if not _catalog_by_id:
        if CATALOG_PATH.exists():
            with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            _catalog_by_id = {r["entity_id"]: CatalogRecord(**r) for r in raw}
        else:
            # Fallback to raw catalog + on-the-fly classification
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
    """Return the catalog keyed by lowercased product name."""
    catalog = get_catalog()
    return {rec.name.lower().strip(): rec for rec in catalog.values()}


def get_index() -> CatalogIndex:
    """Return the pre-built FAISS CatalogIndex (cached after first load)."""
    global _index
    if _index is None:
        _index = CatalogIndex.load(str(INDEX_PATH))
    return _index


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_duration(duration_str: str) -> int:
    """Extract the leading integer from a duration string like '25 minutes'."""
    if not duration_str:
        return 0
    match = re.search(r'\d+', duration_str)
    if match:
        return int(match.group(0))
    return 0
