import os
import requests

from pathlib import Path
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")

API_KEY = os.getenv("TENNIS_DATABASE_API_KEY")

if not API_KEY:
    raise ValueError("Brak klucza Tennis Database API")


URL = "https://api.tennisdatabaseapi.com/v1/matches"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

PARAMS = {
    "date": "2024-06-10",
    "tour": "men",
    "per_page": 2,
}


response = requests.get(
    URL,
    headers=HEADERS,
    params=PARAMS,
    timeout=30,
)

print("HTTP:", response.status_code)

if response.status_code == 200:

    payload = response.json()

    print("Archive: DZIALA")
    print("Meta:", payload.get("meta"))

    for match in payload.get("data", []):

        print(
            match.get("date"),
            (match.get("winner") or {}).get("name"),
            "vs",
            (match.get("loser") or {}).get("name"),
            (match.get("tournament") or {}).get("tier"),
        )

else:

    print("Archive: BRAK DOSTEPU")
    print(response.text[:500])