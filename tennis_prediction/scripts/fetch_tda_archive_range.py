import os
import json
import time
import argparse
import requests

from pathlib import Path
from datetime import date, timedelta
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")

API_KEY = os.getenv("TENNIS_DATABASE_API_KEY")

if not API_KEY:
    raise ValueError("Brak TENNIS_DATABASE_API_KEY")


URL = "https://api.tennisdatabaseapi.com/v1/matches"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

OUTPUT_DIR = ROOT / "data" / "tda_archive"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_day(target_date, max_requests):

    output = OUTPUT_DIR / f"tda_{target_date}.json"

    if output.exists():
        print("POMIJAM, plik istnieje:", target_date)
        return 0, 0

    all_matches = []
    cursor = None
    requests_used = 0

    while True:

        if requests_used >= max_requests:
            raise RuntimeError(
                "Osiagnieto limit requestow dla dnia"
            )

        params = {
            "date": str(target_date),
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

        requests_used += 1

        if response.status_code == 429:

            wait_seconds = int(
                response.headers.get(
                    "Retry-After",
                    60
                )
            )

            print(
                "Limit 120/min. Czekam:",
                wait_seconds,
                "sekund"
            )

            time.sleep(wait_seconds)

            continue

        if response.status_code != 200:

            print(
                "HTTP:",
                response.status_code,
                response.text[:300],
            )

            raise RuntimeError("Blad API")

        payload = response.json()

        matches = payload.get("data", [])
        meta = payload.get("meta", {})

        all_matches.extend(matches)

        print(
            target_date,
            "| request:",
            requests_used,
            "| mecze:",
            len(all_matches),
        )

        cursor = meta.get("next_cursor")

        if not cursor:
            break

        time.sleep(1.0)

    expected = meta.get("total")

    if expected is not None and len(all_matches) != expected:
        raise RuntimeError(
            f"Niepelne dane {len(all_matches)}/{expected}"
        )

    # Zapis dopiero po pobraniu calego dnia
    with open(output, "w", encoding="utf-8") as f:
        json.dump(
            all_matches,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return requests_used, len(all_matches)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)

    parser.add_argument(
        "--max-requests-per-day",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--max-total-requests",
        type=int,
        default=500,
    )

    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    total_requests = 0
    total_matches = 0

    current = start

    while current <= end:

        remaining = (
            args.max_total_requests
            - total_requests
        )

        if remaining <= 0:
            print(
                "STOP: osiagnieto globalny limit API"
            )
            break

        used, count = download_day(
            current,
            min(
                args.max_requests_per_day,
                remaining,
            ),
        )

        total_requests += used
        total_matches += count

        current += timedelta(days=1)

    print("\n===== PODSUMOWANIE =====")
    print("Requesty:", total_requests)
    print("Mecze:", total_matches)
    print("Folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()