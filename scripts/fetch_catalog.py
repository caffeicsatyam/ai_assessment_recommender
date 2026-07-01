"""
One-time script to fetch the SHL product catalog from the remote API.
Run this once to download the initial data file.

Usage:
    python scripts/fetch_catalog.py
"""
import json
import requests

def main():
    url = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
    print(f"Fetching catalog from {url}...")
    
    response = requests.get(url)
    response.raise_for_status()
    
    data = json.loads(response.text, strict=False)
    
    output_file = "data/shl_product_catalog.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    data_info = f"{len(data)} records" if isinstance(data, list) else f"keys: {', '.join(data.keys())}"
    print(f"Successfully saved {data_info} to {output_file}")

if __name__ == "__main__":
    main()
