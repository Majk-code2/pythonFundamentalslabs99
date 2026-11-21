import os
import requests
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv


# Glowny folder projektu tennis_prediction
ROOT = Path(__file__).resolve().parents[1]

# Wczytanie klucza z pliku .env
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("CITO_API_KEY")

if not API_KEY:
    raise ValueError("Brak CITO_API_KEY w pliku .env")


BASE_URL = "https://api.citoapi.com/api/v1/tennis"

HEADERS = {
    "x-api-key": API_KEY
}


# Wczytanie meczow pobranych przez predict_today.py
input_path = ROOT / "data" / "upcoming_atp.csv"

matches = pd.read_csv(input_path)

match_ids = (
    matches["match_id"]
    .dropna()
    .astype(str)
    .tolist()
)

details = []


# Cito przyjmuje maksymalnie 10 identyfikatorow w jednej partii
for start in range(0, len(match_ids), 10):

    batch = match_ids[start:start + 10]

    response = requests.get(
        f"{BASE_URL}/matches",
        headers=HEADERS,
        params={
            "ids": ",".join(batch)
        },
        timeout=30
    )

    print(
        f"Partia {start // 10 + 1}:",
        "HTTP",
        response.status_code
    )

    response.raise_for_status()

    payload = response.json()

    batch_rows = (
        payload.get("data")
        or payload.get("items")
        or []
    )

    details.extend(batch_rows)


print("\n===== SZCZEGOLY MECZOW =====")
print("Pobrane rekordy:", len(details))


# Mapa match_id -> nawierzchnia
surface_map = {}


for match in details:

    match_id = (
        match.get("match_id")
        or match.get("id")
        or match.get("canonical_id")
    )

    tournament = match.get("tournament")

    if not isinstance(tournament, dict):
        tournament = {}

    surface = (
        match.get("surface")
        or match.get("court_surface")
        or tournament.get("surface")
    )

    player1_data = match.get("player1")
    player2_data = match.get("player2")

    if not isinstance(player1_data, dict):
        player1_data = {}

    if not isinstance(player2_data, dict):
        player2_data = {}

    player1 = (
        match.get("player1_name")
        or player1_data.get("name")
        or match.get("winner", {}).get("name")
    )

    player2 = (
        match.get("player2_name")
        or player2_data.get("name")
        or match.get("loser", {}).get("name")
    )

    if match_id:
        surface_map[str(match_id)] = surface

    print("\n--------------------------------------")
    print("Match ID:", match_id)
    print("Player A:", player1)
    print("Player B:", player2)
    print("Surface:", surface)


# Dolaczenie nawierzchni do listy nadchodzacych meczow
matches["match_id"] = matches["match_id"].astype(str)

matches["surface"] = (
    matches["match_id"]
    .map(surface_map)
)


# Zapis wyniku
output_path = (
    ROOT
    / "data"
    / "upcoming_atp_with_surface.csv"
)

matches.to_csv(
    output_path,
    index=False
)


known_surface = matches["surface"].notna().sum()
unknown_surface = matches["surface"].isna().sum()


print("\n===== ZAPIS NAWIERZCHNI =====")
print("Wszystkie mecze:", len(matches))
print("Nawierzchnia znana:", known_surface)
print("Nawierzchnia nieznana:", unknown_surface)
print("Zapisano:", output_path)


print("\n===== MECZE GOTOWE DO MODELU =====")

ready = matches[
    matches["surface"].isin(
        ["Hard", "Clay", "Grass"]
    )
].copy()

columns_to_show = [
    "player_A",
    "player_B",
    "rank_A",
    "rank_B",
    "surface",
    "odds_A",
    "odds_B"
]

print(
    ready[columns_to_show]
    .to_string(index=False)
)

print("\nGotowe mecze:", len(ready))