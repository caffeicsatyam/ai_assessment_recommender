from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATHS = [
    ROOT / "data" / "shl_product_catalog.json",
    ROOT / "shl_product_catalog.json",
]
KEY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}
