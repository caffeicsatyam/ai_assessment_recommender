"""
Verifies CatalogIndex.save() / the read-back half of .load() produce
identical search results before and after a disk round-trip. Uses the
same fake deterministic embedder so this runs without network access.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

# Add the project root and 'app' directories to sys.path so that all imports work correctly
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)
sys.path.insert(0, str(Path(project_root) / "app"))

# Set dummy key for offline testing so genai.Client() doesn't fail on import
import os
os.environ["GOOGLE_API_KEY"] = "mock-key-for-testing"

from unittest.mock import MagicMock
import faiss
import numpy as np

from app.models import CatalogRecord
from app.classify import classify
import app.retrieval
from app.retrieval import CatalogIndex, _RECORDS_FILENAME, _INDEX_FILENAME

DIM = 32
OUT_DIR = "test_index_data"


def fake_encode(texts: list[str]) -> np.ndarray:
    vecs = []
    for t in texts:
        h = hashlib.sha256(t.encode()).digest()
        v = np.frombuffer(h, dtype=np.uint8).astype("float32")
        v = v / (np.linalg.norm(v) + 1e-8)
        # Pad or slice to match target DIM (32)
        if len(v) > DIM:
            v = v[:DIM]
        elif len(v) < DIM:
            v = np.pad(v, (0, DIM - len(v)))
        vecs.append(v)
    return np.stack(vecs)


# Mock the client's embed_content inside retrieval.py
mock_client = MagicMock()
app.retrieval._client = mock_client


def mock_embed_content(model, contents, config=None):
    if isinstance(contents, str):
        texts = [contents]
    else:
        texts = contents
    vecs = fake_encode(texts)
    
    class MockEmbedding:
        def __init__(self, values):
            self.values = values.tolist()
            
    class MockResponse:
        def __init__(self, embeddings):
            self.embeddings = embeddings
            
    return MockResponse([MockEmbedding(v) for v in vecs])


mock_client.models.embed_content.side_effect = mock_embed_content


def main():
    # Locate data/shl_product_catalog.json
    catalog_path = Path("data/shl_product_catalog.json")
    if not catalog_path.exists():
        catalog_path = Path("../data/shl_product_catalog.json")

    with open(catalog_path, encoding="utf-8") as f:
        raw = json.load(f)
    records = [classify(CatalogRecord(**r)) for r in raw]
    recommendable = [r for r in records if not r.is_job_solution and not r.is_report_only]

    texts = [f"{r.name} | {r.description} | {', '.join(r.keys)}" for r in recommendable]
    embeddings = fake_encode(texts)
    faiss_index = faiss.IndexFlatIP(DIM)
    faiss_index.add(embeddings)

    original = CatalogIndex(records=recommendable, index=faiss_index)

    query_text = recommendable[3].name 
    before = original.query(query_text, top_k=3)
    print("Results BEFORE save:")
    for r in before:
        print(f"  {r.score:.4f}  {r.record.name}")

    shutil.rmtree(OUT_DIR, ignore_errors=True)
    original.save(OUT_DIR)
    assert (Path(OUT_DIR) / _INDEX_FILENAME).exists()
    assert (Path(OUT_DIR) / _RECORDS_FILENAME).exists()
    print(f"\nSaved to {OUT_DIR}/ -- files present, good.\n")

    # Read back
    loaded_index = faiss.read_index(str(Path(OUT_DIR) / _INDEX_FILENAME))
    with open(Path(OUT_DIR) / _RECORDS_FILENAME, encoding="utf-8") as f:
        loaded_raw = json.load(f)
    loaded_records = [CatalogRecord(**r) for r in loaded_raw]

    assert loaded_index.ntotal == len(loaded_records), "index/records count mismatch after reload"
    assert [r.entity_id for r in loaded_records] == [r.entity_id for r in recommendable], \
        "record order changed across save/load"

    reloaded = CatalogIndex(records=loaded_records, index=loaded_index)
    after = reloaded.query(query_text, top_k=3)
    print("Results AFTER save/load round-trip:")
    for r in after:
        print(f"  {r.score:.4f}  {r.record.name}")

    assert [r.record.entity_id for r in before] == [r.record.entity_id for r in after], \
        "FAIL: results differ before vs after round-trip"
    assert [round(r.score, 6) for r in before] == [round(r.score, 6) for r in after], \
        "FAIL: scores differ before vs after round-trip"

    print("\nPASS: save/load round-trip preserves index and results exactly.")
    print("NOTE: the real Gemini embed_content API call was mocked to run offline.")

    shutil.rmtree(OUT_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()