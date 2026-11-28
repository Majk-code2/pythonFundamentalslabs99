import os
import json
import requests

from pathlib import Path
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")

API_KEY = os.getenv("TENNIS_DATABASE_API_KEY")

if not API_KEY:
    raise ValueError("Brak klucza TDA")


URL = "https://api.tennisdatabaseapi.com/v1/matches"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}


TARGET_DATE = "2024-06-10"

OUTPUT = (
    ROOT
    / "data"
    / f"tda_archive_{TARGET_DATE}.json"
)


all_matches = []
cursor = None
requests_used = 0


while True:

    params = {
        "date": TARGET_DATE,
        "tour": "men",
        "per_page": 100,
    }

    if cursor:
        params["cursor"] = cursor

    response = requests.get(
        URL,
        headers=HEADERS,
        params=params,
        timeout=30,
    )

    requests_used += 1

    print(
        "Request:",
        requests_used,
        "| HTTP:",
        response.status_code,
    )

    response.raise_for_status()

    payload = response.json()

    matches = payload.get("data", [])
    meta = payload.get("meta", {})

    all_matches.extend(matches)

    print(
        "Pobrano:",
        len(matches),
        "| razem:",
        len(all_matches),
    )

    cursor = meta.get("next_cursor")

    if not cursor:
        break

    if requests_used >= 10:
        raise RuntimeError(
            "Limit bezpieczenstwa: 10 requestow"
        )


expected = meta.get("total")

if expected is not None and len(all_matches) != expected:
    raise RuntimeError(
        f"Niepelne dane: {len(all_matches)}/{expected}"
    )


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    OUTPUT,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        all_matches,
        file,
        ensure_ascii=False,
        indent=2,
    )


print("\n===== PODSUMOWANIE =====")
print("Data:", TARGET_DATE)
print("Requesty:", requests_used)
print("Mecze:", len(all_matches))
print("Zapisano:", OUTPUT)