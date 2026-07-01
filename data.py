import requests
import json

resp = requests.get("https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json")
data = json.loads(resp.text, strict=False)

with open("shl_product_catalog.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(type(data), len(data) if isinstance(data, list) else data.keys())