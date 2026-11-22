import os
import re
import unicodedata
import requests
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv


# Glowny folder projektu tennis_prediction
ROOT = Path(__file__).resolve().parents[1]

# Wczytanie klucza z .env
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("CITO_API_KEY")

if not API_KEY:
    raise ValueError("Brak CITO_API_KEY w pliku .env")


BASE_URL = "https://api.citoapi.com/api/v1/tennis"

HEADERS = {
    "x-api-key": API_KEY
}


ATP_FILES = [
    ROOT / "data" / "2022.csv",
    ROOT / "data" / "2023.csv",
    ROOT / "data" / "2024.csv",
    ROOT / "data" / "2025.csv",
    ROOT / "data" / "2026.csv"
]


CHALLENGER_FILES = [
    ROOT / "data" / "2022_challenger.csv",
    ROOT / "data" / "2023_challenger.csv",
    ROOT / "data" / "2024_challenger.csv",
    ROOT / "data" / "2025_challenger.csv",
    ROOT / "data" / "2026_challenger.csv"
]


def normalize_tournament_name(name):
    """
    Normalizuje nazwe turnieju, aby latwiej laczyc dane Cito
    z historycznymi plikami CSV.
    """

    if not isinstance(name, str):
        return ""

    name = unicodedata.normalize("NFKD", name)

    name = "".join(
        character
        for character in name
        if not unicodedata.combining(character)
    )

    name = name.lower()

    # Usuwa czeste dodatki z nazw Cito
    name = re.sub(
        r"\b(final stage|qualifying|qualification)\b",
        " ",
        name
    )

    # Zostawia litery, cyfry i spacje
    name = re.sub(
        r"[^a-z0-9\s]",
        " ",
        name
    )

    name = re.sub(
        r"\s+",
        " ",
        name
    ).strip()

    return name


def load_tournament_names(files):
    """
    Zwraca zbior znormalizowanych nazw turniejow
    znajdujacych sie w podanych plikach.
    """

    tournament_names = set()

    for file_path in files:

        if not file_path.exists():
            print(
                "Brak pliku historycznego:",
                file_path
            )
            continue

        frame = pd.read_csv(
            file_path,
            usecols=["tourney_name"]
        )

        for name in frame["tourney_name"].dropna():

            normalized = normalize_tournament_name(name)

            if normalized:
                tournament_names.add(normalized)

    return tournament_names


print("===== BUDOWANIE MAPY TURNIEJOW =====")

atp_tournaments = load_tournament_names(
    ATP_FILES
)

challenger_tournaments = load_tournament_names(
    CHALLENGER_FILES
)

print(
    "Turnieje ATP w historii:",
    len(atp_tournaments)
)

print(
    "Turnieje Challenger w historii:",
    len(challenger_tournaments)
)


def detect_competition_level(tournament_name):
    """
    Rozpoznaje ATP lub Challenger na podstawie historycznych plikow.

    Jezeli turniej wystepuje w obu zbiorach albo w zadnym,
    zwraca UNKNOWN zamiast zgadywac.
    """

    normalized = normalize_tournament_name(
        tournament_name
    )

    in_atp = normalized in atp_tournaments
    in_challenger = (
        normalized in challenger_tournaments
    )

    if in_atp and not in_challenger:
        return "ATP"

    if in_challenger and not in_atp:
        return "CHALLENGER"

    # Dodatkowa bezpieczna regola dla Tour Finals
    if (
        "world tour finals" in normalized
        or "atp finals" in normalized
    ):
        return "ATP"

    return "UNKNOWN"


# Wczytanie aktualnych meczow
input_path = (
    ROOT
    / "data"
    / "upcoming_atp.csv"
)

matches = pd.read_csv(input_path)

if matches.empty:
    print("Brak nowych meczow do pobrania nawierzchni.")

    output_path = (
        ROOT
        / "data"
        / "upcoming_atp_with_surface.csv"
    )

    empty_columns = list(matches.columns) + [
        "surface",
        "competition_level",
        "cito_tournament_name"
    ]

    pd.DataFrame(
        columns=empty_columns
    ).to_csv(
        output_path,
        index=False
    )

    print("Zapisano pusty plik:", output_path)
    raise SystemExit(0)

match_ids = (
    matches["match_id"]
    .dropna()
    .astype(str)
    .tolist()
)

details = []


# Cito przyjmuje do 10 identyfikatorow w jednej partii
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


# Mapy danych po match_id
surface_map = {}
level_map = {}
tournament_map = {}


for match in details:

    match_id = (
        match.get("match_id")
        or match.get("id")
        or match.get("canonical_id")
    )

    tournament_object = match.get("tournament")

    if not isinstance(tournament_object, dict):
        tournament_object = {}

    surface = (
        match.get("surface")
        or match.get("court_surface")
        or tournament_object.get("surface")
    )

    tournament_name = (
        match.get("tournament_name")
        or match.get("tournament_display_name")
        or tournament_object.get("name")
        or ""
    )

    competition_level = detect_competition_level(
        tournament_name
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

        match_id = str(match_id)

        surface_map[match_id] = surface
        level_map[match_id] = competition_level
        tournament_map[match_id] = tournament_name

    print("\n--------------------------------------")
    print("Match ID:", match_id)
    print("Turniej Cito:", tournament_name)
    print("Player A:", player1)
    print("Player B:", player2)
    print("Surface:", surface)
    print("Competition level:", competition_level)


# Dolaczenie danych do aktualnych meczow
matches["match_id"] = (
    matches["match_id"]
    .astype(str)
)

matches["surface"] = (
    matches["match_id"]
    .map(surface_map)
)

matches["competition_level"] = (
    matches["match_id"]
    .map(level_map)
)

matches["cito_tournament_name"] = (
    matches["match_id"]
    .map(tournament_map)
)


# Zapis rozszerzonego pliku
output_path = (
    ROOT
    / "data"
    / "upcoming_atp_with_surface.csv"
)

matches.to_csv(
    output_path,
    index=False
)


known_surface = (
    matches["surface"]
    .notna()
    .sum()
)

unknown_surface = (
    matches["surface"]
    .isna()
    .sum()
)

known_level = (
    matches["competition_level"]
    .isin(["ATP", "CHALLENGER"])
    .sum()
)

unknown_level = (
    ~matches["competition_level"]
    .isin(["ATP", "CHALLENGER"])
).sum()


print("\n===== ZAPIS DANYCH =====")
print("Wszystkie mecze:", len(matches))
print("Nawierzchnia znana:", known_surface)
print("Nawierzchnia nieznana:", unknown_surface)
print("Poziom znany:", known_level)
print("Poziom UNKNOWN:", unknown_level)
print("Zapisano:", output_path)


print("\n===== MECZE GOTOWE DO MODELU =====")

ready = matches[
    matches["surface"].isin(
        ["Hard", "Clay", "Grass"]
    )
    & matches["competition_level"].isin(
        ["ATP", "CHALLENGER"]
    )
].copy()


columns_to_show = [
    "player_A",
    "player_B",
    "rank_A",
    "rank_B",
    "surface",
    "competition_level",
    "odds_A",
    "odds_B"
]


if len(ready) > 0:

    print(
        ready[columns_to_show]
        .to_string(index=False)
    )

else:

    print(
        "Brak meczow z rozpoznana nawierzchnia "
        "i poziomem rozgrywek."
    )


print("\nGotowe mecze:", len(ready))


print("\n===== NIEROZPOZNANE TURNIEJE =====")

unknown = matches[
    matches["competition_level"] == "UNKNOWN"
].copy()

if len(unknown) > 0:

    print(
        unknown[
            [
                "tournament",
                "cito_tournament_name",
                "player_A",
                "player_B"
            ]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )

else:

    print("Wszystkie turnieje zostaly rozpoznane.")