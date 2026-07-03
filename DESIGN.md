# Design Documentation

## Design Choices
- **Two-Stage Architecture:** Separated intent/context extraction from the final recommendation and conversational response generation.
- **Strict Intent Classification:** Explicitly categorizing user requests into "search" (new assessments) and "edit" (modifying existing lists). 
- **Structured Generation:** Forced JSON output schemas across all LLM interactions to reliably parse intent, extracted entities, and conversation state (e.g., `end_of_conversation` flags).

## Retrieval Setup
- **Hybrid Search:** Built a custom `CatalogIndex` using `faiss.IndexFlatIP` for dense vector semantic search (`gemini-embedding-001`) alongside a custom BM25-like keyword index.
- **Reciprocal Rank Fusion (RRF):** Merges dense and sparse retrieval results to capture both conceptual matches and exact skill keywords.
- **Query Expansion:** The context extraction LLM expands job roles into targeted technical and soft skills prior to executing the FAISS search.
- **Edit Mode Bypass:** When a user modifies an existing shortlist ("edit" intent), the system bypasses FAISS search and restricts candidates strictly to the previously recommended products.

## Prompt Design
- **Priority-Driven Rules:** System instructions use strictly numbered, prioritized rules (e.g., Rule 0: Off-Topic, Rule 1: Clarification, Rule 2: Edit Rule).
- **Edit Mode Override:** A high-priority `EDIT MODE ACTIVE` block is injected when modifications are detected, strictly prohibiting the LLM from adding or backfilling unrequested items.
- **Authoritative State Tracking:** Injects `[CURRENT_SHORTLIST]` tokens into the conversational history to ground the LLM in the current, factual state of the recommendations.

## Evaluation Approach
- **Conversation Testing:** Manually evaluated against complex multi-turn edge cases (e.g., vague requests, off-topic questions, and granular list modifications).
- **Automated Unit Testing:** Pytest suites verify index persistence (`test_persistence.py`) with offline mock embedders and test robust fallback behaviors for invalid JSON (`test_chat_invalid_json.py`).

## What Didn't Work & Measuring Improvement
- **What Didn't Work:** Initially, standard RAG context failed during "edit" turns. If a user asked to "remove Product X", the LLM would helpfully "fill the gap" with a new, unrequested product from the candidate pool.
- **Measuring Improvement:** We measured improvement by tracking the retention of untouched products during "edit" turns. By restricting candidate products entirely to previously suggested items and enforcing strict prompt constraints during edits, we achieved consistent compliance without hallucinated additions.

## AI Tools Used
- **Agentic Coding (Google DeepMind Antigravity / Gemini 3.1 Pro):** Used as an AI pair-programmer to rapidly debug and implement solutions for complex prompt compliance—specifically, designing the explicit "edit" vs "search" intent logic and strictly enforcing shortlist retention.
