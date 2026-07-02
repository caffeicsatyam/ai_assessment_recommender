from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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


_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "to", "for", "is", "it",
    "on", "at", "by", "with", "as", "from", "that", "this", "are", "was",
    "be", "has", "have", "not", "but", "can", "will", "do", "if", "its",
    "all", "no", "so", "up", "out", "one", "new", "also", "about", "which",
})


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop words."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


logger = logging.getLogger(__name__)


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
        # Build keyword inverted index for hybrid retrieval
        self._inv_index: dict[str, list[int]] = defaultdict(list)
        self._doc_lens: list[int] = []
        self._avg_dl: float = 0.0
        self._build_keyword_index()

    def _build_keyword_index(self) -> None:
        """Build an inverted index over record texts for keyword search."""
        total_len = 0
        for i, rec in enumerate(self.records):
            tokens = _tokenize(_embedding_text(rec))
            self._doc_lens.append(len(tokens))
            total_len += len(tokens)
            for token in set(tokens):  # unique tokens per doc
                self._inv_index[token].append(i)
        self._avg_dl = total_len / max(len(self.records), 1)

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
    def query(self, text: str, top_k: int = 20, min_score: float = 0.40) -> list[RetrievalResult]:
        """
        Recall-biased semantic search. Returns up to top_k results ordered
        by descending similarity, filtering out any result below min_score
        to avoid surfacing irrelevant candidates.
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
            if idx != -1 and float(score) >= min_score
        ]
        return results

    def multi_query(self, texts: list[str], top_k_each: int = 10, max_total: int = 30) -> list[RetrievalResult]:
        """
        Runs multiple sub-queries (e.g. decomposed by skill / role-level /
        soft-skill angle) and merges results, deduplicating by entity_id
        and keeping the highest score seen for each record. Caps output
        at max_total to avoid flooding the LLM with low-relevance noise.
        """
        best: dict[str, RetrievalResult] = {}
        for text in texts:
            for r in self.query(text, top_k=top_k_each):
                key = r.record.entity_id
                if key not in best or r.score > best[key].score:
                    best[key] = r
        ranked = sorted(best.values(), key=lambda r: r.score, reverse=True)
        return ranked[:max_total]

    # ------------------------------------------------------------------
    # Keyword search (BM25-lite, no external deps)
    # ------------------------------------------------------------------
    def keyword_query(self, text: str, top_k: int = 20) -> list[RetrievalResult]:
        """
        BM25-lite keyword search. Scores each document by term-frequency
        overlap with the query, adjusted for document length.
        Complements the semantic search by catching exact term matches
        that embeddings might miss or underweight.
        """
        query_tokens = _tokenize(text)
        if not query_tokens:
            return []

        k1, b = 1.5, 0.75  # standard BM25 parameters
        n_docs = len(self.records)
        scores: dict[int, float] = defaultdict(float)

        for token in query_tokens:
            posting = self._inv_index.get(token, [])
            if not posting:
                continue
            idf = math.log((n_docs - len(posting) + 0.5) / (len(posting) + 0.5) + 1.0)
            for doc_idx in posting:
                # Count term frequency in this document
                doc_tokens = _tokenize(_embedding_text(self.records[doc_idx]))
                tf = doc_tokens.count(token)
                dl = self._doc_lens[doc_idx]
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self._avg_dl))
                scores[doc_idx] += idf * tf_norm

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievalResult(record=self.records[idx], score=score)
            for idx, score in ranked
            if score > 0
        ]

    # ------------------------------------------------------------------
    # Hybrid retrieval: semantic + keyword via Reciprocal Rank Fusion
    # ------------------------------------------------------------------
    def hybrid_multi_query(
        self,
        texts: list[str],
        top_k_each: int = 10,
        max_total: int = 30,
        rrf_k: int = 60,
    ) -> list[RetrievalResult]:
        """
        Combines semantic (FAISS) and keyword (BM25) retrieval using
        Reciprocal Rank Fusion (RRF). For each sub-query, both retrieval
        methods contribute ranked lists that are merged with:
            RRF_score(doc) = sum( 1 / (rrf_k + rank) )
        This balances dense embedding recall with exact keyword precision.
        """
        # Collect per-entity_id: best record + RRF score
        rrf_scores: dict[str, float] = defaultdict(float)
        best_record: dict[str, CatalogRecord] = {}

        for text in texts:
            # Semantic results
            sem_results = self.query(text, top_k=top_k_each)
            for rank, r in enumerate(sem_results):
                eid = r.record.entity_id
                rrf_scores[eid] += 1.0 / (rrf_k + rank + 1)
                best_record[eid] = r.record

            # Keyword results
            kw_results = self.keyword_query(text, top_k=top_k_each)
            for rank, r in enumerate(kw_results):
                eid = r.record.entity_id
                rrf_scores[eid] += 1.0 / (rrf_k + rank + 1)
                best_record[eid] = r.record

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:max_total]
        logger.info(
            "Hybrid retrieval: %d unique candidates from %d queries.",
            len(ranked), len(texts),
        )
        return [
            RetrievalResult(record=best_record[eid], score=score)
            for eid, score in ranked
        ]