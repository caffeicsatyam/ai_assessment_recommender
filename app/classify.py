from __future__ import annotations

import re
from .models import CatalogRecord, KEYS_TO_CODE


# ---------------------------------------------------------------------------
# is_job_solution()
# ---------------------------------------------------------------------------

_SOLUTION_NAME_RE = re.compile(r"\bsolution\b", re.IGNORECASE)
_FOCUS_SERIES_RE = re.compile(r"\bfocus\s*8\.0\b", re.IGNORECASE)
_ROLE_FRAMING_RE = re.compile(
    r"(is for (entry-level|a wide range of)|sample tasks for these jobs include|"
    r"potential job titles that use this (solution|simulation))",
    re.IGNORECASE,
)


def is_job_solution(record: CatalogRecord) -> bool:
    name = record.name or ""
    description = record.description or ""

    if _SOLUTION_NAME_RE.search(name):
        return True

    if _FOCUS_SERIES_RE.search(name):
        return True

    if _ROLE_FRAMING_RE.search(description) and len(record.keys) >= 2:
        return True

    return False


# ---------------------------------------------------------------------------
# is_report_only()
# ---------------------------------------------------------------------------

_REPORT_NAME_RE = re.compile(r"\breport\b", re.IGNORECASE)

_BLANK_DURATION = ""


def is_report_only(record: CatalogRecord) -> bool:
    """
    True if this record is a report derived from an assessment (not an
    assessment a candidate takes). Reports are TAGGED, not deleted, from
    the catalog -- see note below.
    """
    name = record.name or ""
    if _REPORT_NAME_RE.search(name) and (record.duration or "") == _BLANK_DURATION:
        return True
    return False


# ---------------------------------------------------------------------------
# test_type derivation
# ---------------------------------------------------------------------------

def derive_test_type(record: CatalogRecord) -> str:
    """
    Map record.keys -> comma-joined single-letter codes, e.g. "K,S".
    Order follows KEYS_TO_CODE's canonical ordering, not the raw JSON's
    (sometimes inconsistent) key ordering, so output is deterministic.
    """
    codes = [KEYS_TO_CODE[k] for k in KEYS_TO_CODE if k in record.keys]
    return ",".join(codes)


def classify(record: CatalogRecord) -> CatalogRecord:
    """Populate all derived fields on a record in place and return it."""
    record.is_job_solution = is_job_solution(record)
    record.is_report_only = is_report_only(record)
    record.test_type = derive_test_type(record)
    return record


# ---------------------------------------------------------------------------
# NOTE on why reports are tagged, not deleted:
#
# The default recommendation candidate pool should exclude is_report_only
# records (a candidate is never "assigned" a report to take). But deleting
# them from the underlying catalog would make it impossible to answer a
# legitimate compare/info question like "what report comes with the OPQ32r?"
# if such a question ever appears in a holdout trace. Filtering happens at
# the retrieval-candidate-pool level, not at the data-loading level.
# ---------------------------------------------------------------------------