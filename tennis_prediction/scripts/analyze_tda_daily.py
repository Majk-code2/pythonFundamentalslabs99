import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "tda_2026-11-23.json"
)


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8",
) as file:
    matches = json.load(file)


print("===== TENNIS DATABASE API =====")
print("Mecze:", len(matches))


tiers = Counter()
surfaces = Counter()
players = {}
qualifications = 0
closing_odds = 0


for match in matches:

    tournament = (
        match.get("tournament")
        or {}
    )

    tier = tournament.get(
        "tier",
        "UNKNOWN",
    )

    surface = tournament.get(
        "surface",
        "UNKNOWN",
    )

    tournament_name = str(
        tournament.get("name", "")
    )

    tiers[tier] += 1
    surfaces[surface] += 1

    if "qual" in tournament_name.lower():
        qualifications += 1

    odds = match.get("odds") or {}

    if odds.get("closing"):
        closing_odds += 1

    for side in ["winner", "loser"]:

        player = match.get(side) or {}

        player_id = player.get("id")
        player_name = player.get("name")

        if player_id:

            players[player_id] = {
                "name": player_name,
                "country": player.get("country"),
            }


print("\n===== POZIOMY =====")
for key, value in tiers.most_common():
    print(key, ":", value)


print("\n===== NAWIERZCHNIE =====")
for key, value in surfaces.most_common():
    print(key, ":", value)


print("\n===== POKRYCIE =====")
print("Kwalifikacje:", qualifications)
print("Closing odds:", closing_odds)
print("Unikalni zawodnicy:", len(players))


SEARCH_NAMES = [
    "arias",
    "loureiro",
    "tosetto",
    "rodrigues",
    "schenekenberg",
]


print("\n===== SZUKANI ZAWODNICY =====")

for search_name in SEARCH_NAMES:

    found = []

    for player_id, player in players.items():

        player_name = str(
            player.get("name", "")
        ).lower()

        if search_name in player_name:

            found.append(
                (
                    player_id,
                    player["name"],
                    player["country"],
                )
            )

    print(
        search_name,
        ":",
        found if found else "BRAK",
    )


print("\n===== TURNIEJE =====")

tournaments = Counter(
    (
        (match.get("tournament") or {})
        .get("name", "UNKNOWN")
    )
    for match in matches
)

for name, count in tournaments.most_common():
    print(count, "|", name)