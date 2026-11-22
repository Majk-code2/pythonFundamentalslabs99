import os
import requests
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")

API_KEY = os.getenv("CITO_API_KEY")

if not API_KEY:
    raise ValueError("Brak CITO_API_KEY w pliku .env")

BASE_URL = "https://api.citoapi.com/api/v1/tennis"

HEADERS = {
    "x-api-key": API_KEY
}


def get_json(path, params=None):
    response = requests.get(
        BASE_URL + path,
        headers=HEADERS,
        params=params,
        timeout=30
    )

    response.raise_for_status()
    return response.json()


# Pobieramy bieżące wydarzenia
odds_data = get_json("/odds/upcoming")

items = odds_data.get("items", [])

# Tylko nierozpoczęte single ATP z normalnymi ID zawodników
matches = []

for match in items:

    home = match.get("home", {})
    away = match.get("away", {})

    home_id = home.get("player_id", "")
    away_id = away.get("player_id", "")

    if match.get("live", False):
        continue

    if match.get("tour") != "ATP":
        continue

    if not home_id.startswith("atp_"):
        continue

    if not away_id.startswith("atp_"):
        continue

    # Pomijamy identyfikatory par deblowych i zastępcze
    if home_id.startswith("atp_p_"):
        continue

    if away_id.startswith("atp_p_"):
        continue

    if "/" in home.get("name", ""):
        continue

    if "/" in away.get("name", ""):
        continue

    matches.append(match)


print("===== NIERozPOCZETE SINGLE ATP =====")
print("Znalezione mecze:", len(matches))


# Pobieramy profile wyłącznie dla zawodników z tych meczów
player_ids = sorted(
    {
        player_id
        for match in matches
        for player_id in [
            match["home"]["player_id"],
            match["away"]["player_id"]
        ]
    }
)

profiles = {}

for player_id in player_ids:

    try:
        profile = get_json(f"/players/{player_id}")

        profiles[player_id] = {
            "name": profile.get("full_name"),
            "rank": (
                profile.get("current_rank")
                or profile.get("last_rank")
            )
        }

    except requests.HTTPError as error:
        print(
            "Nie udalo sie pobrac profilu:",
            player_id,
            error
        )


print("\n===== MECZE Z RANKINGAMI =====")

usable_matches = []

for match in matches:

    home_id = match["home"]["player_id"]
    away_id = match["away"]["player_id"]

    home_profile = profiles.get(home_id, {})
    away_profile = profiles.get(away_id, {})

    rank_a = home_profile.get("rank")
    rank_b = away_profile.get("rank")

    if rank_a is None or rank_b is None:
        continue

    quotes = match.get("quotes_by_bookmaker", {})
    bookmaker = match.get("bookmaker")
    h2h = quotes.get(bookmaker, {}).get("h2h", {})

    row = {
        "match_id": match.get("match_id"),
        "start": match.get("commence_time"),
        "tournament": match.get("tournament"),
        "player_A": home_profile.get("name"),
        "player_B": away_profile.get("name"),
        "rank_A": rank_a,
        "rank_B": rank_b,
        "odds_A": h2h.get("player1_odds"),
        "odds_B": h2h.get("player2_odds")
    }

    usable_matches.append(row)

    print("\n--------------------------------------")
    print("Turniej:", row["tournament"])
    print("Start:", row["start"])
    print(
        f"{row['player_A']} (#{row['rank_A']})"
        f" vs "
        f"{row['player_B']} (#{row['rank_B']})"
    )
    print("Kurs A:", row["odds_A"])
    print("Kurs B:", row["odds_B"])


output_columns = [
    "match_id",
    "start",
    "tournament",
    "player_A",
    "player_B",
    "rank_A",
    "rank_B",
    "odds_A",
    "odds_B"
]

output = pd.DataFrame(
    usable_matches,
    columns=output_columns
)

output.to_csv(
    ROOT / "data" / "upcoming_atp.csv",
    index=False
)

print("\n===== PODSUMOWANIE =====")
print("Mecze gotowe do dalszej analizy:", len(output))
print("Zapisano: data/upcoming_atp.csv")