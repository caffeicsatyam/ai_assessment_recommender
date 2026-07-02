from __future__ import annotations
import re
from .models import CatalogRecord, KEYS_TO_CODE

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


_REPORT_NAME_RE = re.compile(r"\breport\b", re.IGNORECASE)
_BLANK_DURATION = ""


def is_report_only(record: CatalogRecord) -> bool:
    name = record.name or ""
    if _REPORT_NAME_RE.search(name) and (record.duration or "") == _BLANK_DURATION:
        return True
    return False


def derive_test_type(record: CatalogRecord) -> str:
    codes = [KEYS_TO_CODE[k] for k in KEYS_TO_CODE if k in record.keys]
    return ",".join(codes)


def classify(record: CatalogRecord) -> CatalogRecord:
    record.is_job_solution = is_job_solution(record)
    record.is_report_only = is_report_only(record)
    record.test_type = derive_test_type(record)
    return record
