import os
import json
import requests

from pathlib import Path
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")

API_KEY = os.getenv("TENNIS_DATABASE_API_KEY")

if not API_KEY:
    raise ValueError(
        "Brak TENNIS_DATABASE_API_KEY"
    )


URL = "https://api.tennisdatabaseapi.com/v1/matches"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}


# Automatycznie wybieramy wczorajsza date UTC
today_utc = datetime.now(timezone.utc).date()

target_date = (
    today_utc - timedelta(days=1)
).isoformat()


OUTPUT_FILE = (
    ROOT
    / "data"
    / f"tda_{target_date}.json"
)


print("===== TENNIS DATABASE API =====")
print("Pobierana data:", target_date)


# Nie pobieramy ponownie istniejacego pliku
if OUTPUT_FILE.exists():

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        existing_data = json.load(file)

    if isinstance(existing_data, list) and existing_data:
        print(
            "Dane za ten dzien juz istnieja:",
            len(existing_data),
            "meczow"
        )

        print("Zapisano:", OUTPUT_FILE)
        raise SystemExit(0)


all_matches = []
cursor = None
request_count = 0

MAX_REQUESTS = 10


while True:

    params = {
        "date": target_date,
        "tour": "men",
        "per_page": 50,
    }

    if cursor:
        params["cursor"] = cursor

    response = requests.get(
        URL,
        headers=HEADERS,
        params=params,
        timeout=30,
    )

    request_count += 1

    print(
        "Request:",
        request_count,
        "| HTTP:",
        response.status_code
    )

    if response.status_code != 200:

        print(
            "Blad API:",
            response.text[:500]
        )

        raise SystemExit(1)


    payload = response.json()

    data = payload.get("data", [])
    meta = payload.get("meta", {})

    all_matches.extend(data)

    print(
        "Pobrano:",
        len(data),
        "| razem:",
        len(all_matches)
    )

    cursor = meta.get("next_cursor")

    if not cursor:
        break

    if request_count >= MAX_REQUESTS:

        raise RuntimeError(
            "Osiagnieto limit requestow. "
            "Nie zapisano niepelnego dnia."
        )


# Kontrola kompletnosci
expected_total = meta.get("total")

if (
    expected_total is not None
    and len(all_matches) != expected_total
):
    raise RuntimeError(
        f"Niepelne dane: {len(all_matches)} "
        f"z {expected_total}"
    )


# Tworzymy katalog, jesli nie istnieje
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# Zapisujemy dopiero po pobraniu wszystkich stron
with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_matches,
        file,
        ensure_ascii=False,
        indent=2
    )


print("\n===== PODSUMOWANIE =====")
print("Data:", target_date)
print("Requesty:", request_count)
print("Mecze:", len(all_matches))


tiers = {}

for match in all_matches:

    tournament = (
        match.get("tournament") or {}
    )

    tier = tournament.get(
        "tier",
        "UNKNOWN"
    )

    tiers[tier] = (
        tiers.get(tier, 0) + 1
    )


print("Poziomy:", tiers)


qualification = [
    match
    for match in all_matches
    if "qual" in str(
        (match.get("tournament") or {})
        .get("name", "")
    ).lower()
]


print(
    "Kwalifikacje:",
    len(qualification)
)

print("\nZapisano:")
print(OUTPUT_FILE)