import json
import os
from pathlib import Path
from typing import Optional
from google import genai
from google.genai import types
import re

try:
    from .models import CatalogRecord
    from .retrieval import CatalogIndex
except ImportError:
    from app.models import CatalogRecord
    from app.retrieval import CatalogIndex

# Client lazy initialization
_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv(Path(__file__).resolve().parents[1] / ".env")
                api_key = os.environ.get("GOOGLE_API_KEY")
            except ImportError:
                pass
        _client = genai.Client(api_key=api_key)
    return _client

#
CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog_clean.json"
INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "index_data"

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
            # Fallback
            raw_path = Path(__file__).resolve().parent.parent / "data" / "shl_product_catalog.json"
            if raw_path.exists():
                with open(raw_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                from classify import classify
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
    match = re.search(r'\d+', duration_str)
    if match:
        return int(match.group(0))
    return 0

def make_recommendation(record: CatalogRecord, reason: str) -> dict:
    return {
        "id": record.entity_id,
        "name": record.name,
        "url": record.link,
        "test_type": record.test_type or "K",
        "duration": parse_duration(record.duration),
        "remote_testing": record.remote.lower() == "yes" if record.remote else True,
        "reason": reason
    }

def extract_context(messages: list[dict]) -> tuple[list[str], list[str]]:
    convo = ""
    for msg in messages:
        convo += f"{msg['role'].upper()}: {msg['content']}\n"
        
    prompt = f"""You are analyzing a conversation between a user and an AI recruiter hiring coordinator.
Based on the conversation history below:
1. Extract 1 to 3 search queries (key skills, technologies, or job roles) to find new candidate assessments in the catalog.
2. Identify and extract the exact `entity_id` or product names of any SHL assessments that were recommended/shortlisted by the assistant in the previous turns.

Conversation:
{convo}

Output as JSON with keys:
- "search_queries": list of strings (queries for searching catalog)
- "previously_recommended_ids": list of strings (entity_ids or names of previously recommended products)
"""
    try:
        client = get_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "search_queries": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"}
                        },
                        "previously_recommended_ids": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"}
                        }
                    },
                    "required": ["search_queries", "previously_recommended_ids"]
                }
            )
        )
        res = json.loads(response.text)
        queries = res.get("search_queries", [])
        prev_ids = res.get("previously_recommended_ids", [])
        return queries, prev_ids
    except Exception as e:
        print(f"Error extracting context: {e}")
        return [messages[-1]["content"]], []

def get_candidates(queries: list[str], prev_ids_or_names: list[str]) -> list[CatalogRecord]:
    index = get_index()
    catalog = get_catalog()
    catalog_by_name = get_catalog_by_name()
    
    # 1. Fetch from search queries
    results = index.multi_query(queries, top_k_each=15)
    candidates = [r.record for r in results]
    
    candidate_ids = {rec.entity_id for rec in candidates}
    for item in prev_ids_or_names:
        item_clean = item.strip()
        # Try matching by ID
        if item_clean in catalog:
            rec = catalog[item_clean]
            if rec.entity_id not in candidate_ids:
                candidates.append(rec)
                candidate_ids.add(rec.entity_id)
        else:

            rec = catalog_by_name.get(item_clean.lower())
            if rec and rec.entity_id not in candidate_ids:
                candidates.append(rec)
                candidate_ids.add(rec.entity_id)
            else:
                for name, rec in catalog_by_name.items():
                    if item_clean.lower() in name or name in item_clean.lower():
                        if rec.entity_id not in candidate_ids:
                            candidates.append(rec)
                            candidate_ids.add(rec.entity_id)
                            break
                            
    return candidates

def format_candidates(candidates: list[CatalogRecord]) -> str:
    parts = []
    for rec in candidates:
        keys_str = ", ".join(rec.keys)
        parts.append(
            f"ID: {rec.entity_id}\n"
            f"Name: {rec.name}\n"
            f"Test Type: {rec.test_type}\n"
            f"Keys: {keys_str}\n"
            f"Duration: {rec.duration}\n"
            f"Languages: {rec.languages_raw}\n"
            f"Description: {rec.description}\n"
            f"URL: {rec.link}\n"
            f"---"
        )
    return "\n".join(parts)

def generate_response(messages: list[dict]) -> tuple[str, list[dict], bool]:
    # 1. Extract queries and previously recommended products
    queries, prev_items = extract_context(messages)
    
    # 2. Retrieve candidates
    candidates = get_candidates(queries, prev_items)
    candidates_str = format_candidates(candidates)
    
    # 3. Call main Gemini model for routing & recommendation
    convo = ""
    for msg in messages:
        convo += f"{msg['role'].upper()}: {msg['content']}\n"
        
    system_instruction = """You are an expert AI recruiting and assessment coordinator recommending assessments from the SHL product catalog.
Your goal is to build a shortlist of 1 to 10 assessments matching the hiring manager's requirements.

Follow these strict rules:
1. CLARIFICATION RULE:
   - If the user has NOT provided enough context to make a specific recommendation (e.g. they haven't specified the role seniority/level, language requirements, accent preferences for voice tests, or what specific skills to assess), do NOT recommend any products yet.
   - Instead, ask 1 or 2 targeted clarifying questions to narrow down the requirements. Set `recommendations` to an empty list `[]` and `end_of_conversation` to `false`.
   - Examples of clarifying: "What language are the calls in?", "Is the seniority level closer to a senior IC or a tech lead?".

2. SHORTLIST SELECTION RULE:
   - When you have enough context, select a list of 1 to 10 appropriate assessments from the Candidate Products list.
   - For each recommended product, write a clear, personalized explanation (`reason`) of why it fits this role and level (e.g., "Advanced covers concurrency and JVM internals which fits a senior microservices engineer").
   - You can ONLY recommend products that are explicitly listed in the Candidate Products. Do not hallucinate product names or IDs.
   - Maintain state: if the user asks to add or drop products, modify the previous list of recommended products accordingly. Keep other recommended products unless they were dropped.

3. CONVERSATION END RULE:
   - When the user confirms they are satisfied with the shortlist (e.g., "Perfect", "That works", "Confirmed", "Locking it in"), set `end_of_conversation` to `true` and output the final shortlist.
   - Otherwise, set `end_of_conversation` to `false`.

4. OUTPUT FORMAT:
   - You must output valid JSON matching the schema below.
   - Do NOT include markdown tables in your conversational `reply` text. Keep the `reply` conversational.
"""

    prompt = f"""Candidate Products:
        {candidates_str}

        Conversation History:
        {convo}

        Please analyze the conversation and output your decision as JSON with the following schema:
        {{
        "reply": "Conversational reply text to the user",
        "recommendations": [
            {{
            "entity_id": "product entity ID",
            "reason": "personalized explanation of why this product fits"
            }}
        ],
        "end_of_conversation": true/false
}}
"""

    try:
        client = get_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "reply": {"type": "STRING"},
                        "end_of_conversation": {"type": "BOOLEAN"},
                        "recommendations": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "entity_id": {"type": "STRING"},
                                    "reason": {"type": "STRING"}
                                },
                                "required": ["entity_id", "reason"]
                            }
                        }
                    },
                    "required": ["reply", "end_of_conversation", "recommendations"]
                }
            )
        )
        
        res = json.loads(response.text)
        reply = res.get("reply", "")
        end_of_conversation = res.get("end_of_conversation", False)
        rec_items = res.get("recommendations", [])
        
        catalog = get_catalog()
        recommendations = []
        for item in rec_items:
            eid = item.get("entity_id")
            reason = item.get("reason", "")
            if eid in catalog:
                rec_record = catalog[eid]
                rec_dict = make_recommendation(rec_record, reason)
                recommendations.append(rec_dict)
                
        return reply, recommendations, end_of_conversation
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I encountered an error while processing your request. Please try again.", [], False
