import os
import json
import requests

from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

api_key = os.getenv("CITO_API_KEY")

if not api_key:
    raise ValueError("Brak CITO_API_KEY w pliku .env")

BASE_URL = "https://api.citoapi.com/api/v1/tennis"

headers = {
    "x-api-key": api_key
}

# ID zawodnikow z odpowiedzi Cito
player_ids = [
    "atp_207989",  # Carlos Alcaraz
    "atp_210097"   # Ben Shelton
]

for player_id in player_ids:

    url = f"{BASE_URL}/players/{player_id}"

    response = requests.get(
        url,
        headers=headers,
        timeout=30
    )

    print("\n===================================")
    print("Player ID:", player_id)
    print("Status HTTP:", response.status_code)

    response.raise_for_status()

    data = response.json()

    print(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        )[:3000]
    )