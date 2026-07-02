from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

try:
    from .llm_client import get_gemini_client as _get_client
except ImportError:
    from app.llm_client import get_gemini_client as _get_client

try:
    from .models import CatalogRecord
except ImportError:
    from app.models import CatalogRecord

_MODEL_NAME = "models/gemini-embedding-001"
_INDEX_FILENAME = "catalog.index"
_RECORDS_FILENAME = "catalog_indexed.json"


def _embedding_text(record: CatalogRecord) -> str:
    """Build the text that gets embedded for a single catalog record."""
    keys_str = ", ".join(record.keys)
    parts = [record.name, record.description, keys_str]
    return " | ".join(p for p in parts if p)


@dataclass
class RetrievalResult:
    record: CatalogRecord
    score: float  


class CatalogIndex:
    """
    Wraps a FAISS index + the underlying catalog records for semantic
    search. Only records where is_job_solution=False and
    is_report_only=False are included.

    Construct via CatalogIndex.build(...) ONCE (offline, e.g. in a build
    script), then CatalogIndex.save(dir). At runtime, use
    CatalogIndex.load(dir) -- this skips re-embedding the catalog and
    only loads the model for encoding incoming queries.
    """

    def __init__(self, records: list[CatalogRecord], index: faiss.Index):
        self.records = records
        self.index = index

    # ------------------------------------------------------------------
    # One-time build (offline step)
    # ------------------------------------------------------------------
    @classmethod
    def build(cls, records: list[CatalogRecord], model_name: str = _MODEL_NAME) -> "CatalogIndex":
        """
        Embeds the recommendable subset of `records` and builds a FAISS
        index from scratch. Run this ONCE, offline, then call .save() to
        persist -- do not call this at server startup.
        """
        recommendable = [r for r in records if not r.is_job_solution and not r.is_report_only]
        if not recommendable:
            raise ValueError(
                "No recommendable records found -- check that classify() was "
                "run on this catalog before building the index."
            )

        texts = [_embedding_text(r) for r in recommendable]

        _BATCH_SIZE = 100
        all_embeddings = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            while True:
                try:
                    response = _get_client().models.embed_content(
                        model=model_name,
                        contents=batch,
                        config={"task_type": "RETRIEVAL_DOCUMENT"},
                    )
                    all_embeddings.extend([e.values for e in response.embeddings])
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        print(f"Rate limit hit at batch starting at {i}. Sleeping 60s before retry...")
                        time.sleep(60)
                    else:
                        raise e

        embeddings = np.asarray(all_embeddings, dtype="float32")
        row_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / row_norms

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        return cls(records=recommendable, index=index)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, dir_path: str) -> None:
        """Persist the FAISS index and the exact record ordering it was built
        from. Record order MUST match embedding order -- do not re-sort
        records.json independently of the index."""
        out_dir = Path(dir_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(out_dir / _INDEX_FILENAME))
        with open(out_dir / _RECORDS_FILENAME, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in self.records], f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, dir_path: str, model_name: str = _MODEL_NAME) -> "CatalogIndex":
        """
        Fast path for server startup: reads the pre-built FAISS index and
        record list from disk. Only loads the embedding model itself
        (needed to embed incoming query text each turn) -- does NOT
        re-embed the ~300 catalog texts. This is the method the FastAPI
        service should call on startup.
        """
        in_dir = Path(dir_path)
        index = faiss.read_index(str(in_dir / _INDEX_FILENAME))
        with open(in_dir / _RECORDS_FILENAME, "r", encoding="utf-8") as f:
            raw = json.load(f)
        records = [CatalogRecord(**r) for r in raw]

        if index.ntotal != len(records):
            raise ValueError(
                f"Index/records mismatch: index has {index.ntotal} vectors "
                f"but records file has {len(records)} entries. Rebuild via "
                f"CatalogIndex.build().save()."
            )

        return cls(records=records, index=index)

    # ------------------------------------------------------------------
    # Query (runtime, per-turn -- only the query text gets embedded here)
    # ------------------------------------------------------------------
    def query(self, text: str, top_k: int = 20) -> list[RetrievalResult]:
        """
        Recall-biased semantic search. Returns up to top_k results ordered
        by descending similarity. Callers (router logic) are responsible
        for further narrowing to the final 1-10 shown to the user, applying
        structured constraints (job_level, language, etc.) along the way.
        """
        top_k = min(top_k, len(self.records))
        response = _get_client().models.embed_content(
            model=_MODEL_NAME,
            contents=text,
            config={"task_type": "RETRIEVAL_QUERY"},
        )
        query_vec = np.asarray(
            [response.embeddings[0].values], dtype="float32"
        )
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

        scores, indices = self.index.search(query_vec, top_k)
        results = [
            RetrievalResult(record=self.records[idx], score=float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]
        return results

    def multi_query(self, texts: list[str], top_k_each: int = 10) -> list[RetrievalResult]:
        """
        Runs multiple sub-queries (e.g. decomposed by skill / role-level /
        soft-skill angle) and merges results, deduplicating by entity_id
        and keeping the highest score seen for each record. This directly
        targets recall: a single combined query can under-retrieve when a
        persona has several distinct facets (e.g. "Java developer" +
        "works with stakeholders" are two different semantic clusters).
        """
        best: dict[str, RetrievalResult] = {}
        for text in texts:
            for r in self.query(text, top_k=top_k_each):
                key = r.record.entity_id
                if key not in best or r.score > best[key].score:
                    best[key] = r
        return sorted(best.values(), key=lambda r: r.score, reverse=True)