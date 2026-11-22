import pandas as pd

from pathlib import Path
from collections import defaultdict

from player_name_utils import normalize_player_name


ROOT = Path(__file__).resolve().parents[1]

START_ELO = 1500.0
K_FACTOR = 24.0


FILES = [
    ROOT / "data" / "2022.csv",
    ROOT / "data" / "2022_challenger.csv",
    ROOT / "data" / "2023.csv",
    ROOT / "data" / "2023_challenger.csv",
    ROOT / "data" / "2024.csv",
    ROOT / "data" / "2024_challenger.csv",
    ROOT / "data" / "2025.csv",
    ROOT / "data" / "2025_challenger.csv",
    ROOT / "data" / "2026.csv",
    ROOT / "data" / "2026_challenger.csv",
]


ROUND_ORDER = {
    "Q1": 1,
    "Q2": 2,
    "Q3": 3,
    "R128": 4,
    "R64": 5,
    "R32": 6,
    "R16": 7,
    "QF": 8,
    "SF": 9,
    "F": 10,
    "BR": 11,
}


def expected_score(rating_a, rating_b):
    return 1.0 / (
        1.0
        + 10.0 ** (
            (rating_b - rating_a) / 400.0
        )
    )


data = pd.concat(
    [pd.read_csv(file) for file in FILES],
    ignore_index=True,
)


data["tourney_date"] = pd.to_datetime(
    data["tourney_date"].astype(str),
    format="%Y%m%d",
    errors="coerce",
)


data = data.dropna(
    subset=[
        "tourney_date",
        "tourney_id",
        "surface",
        "round",
        "winner_name",
        "loser_name",
    ]
).copy()


data = data[
    data["surface"].isin(
        ["Hard", "Clay", "Grass"]
    )
].copy()


data["round_order"] = (
    data["round"].map(ROUND_ORDER)
)


data = data.dropna(
    subset=["round_order"]
).copy()


data["round_order"] = (
    data["round_order"].astype(int)
)


data["winner_key"] = (
    data["winner_name"]
    .apply(normalize_player_name)
)


data["loser_key"] = (
    data["loser_name"]
    .apply(normalize_player_name)
)


data = data.sort_values(
    [
        "tourney_date",
        "tourney_id",
        "round_order",
        "match_num",
    ]
).reset_index(drop=True)


overall_elo = defaultdict(
    lambda: START_ELO
)


surface_elo = defaultdict(
    lambda: defaultdict(
        lambda: START_ELO
    )
)


display_names = {}


for (
    tourney_date,
    tourney_id,
), tournament in data.groupby(
    [
        "tourney_date",
        "tourney_id",
    ],
    sort=True,
):

    for _, round_matches in tournament.groupby(
        "round_order",
        sort=True,
    ):

        round_updates = []

        for _, match in round_matches.iterrows():

            winner = match["winner_key"]
            loser = match["loser_key"]
            surface = match["surface"]

            display_names[winner] = match["winner_name"]
            display_names[loser] = match["loser_name"]

            round_updates.append(
                (
                    winner,
                    loser,
                    surface,
                )
            )

        for winner, loser, surface in round_updates:

            winner_rating = overall_elo[winner]
            loser_rating = overall_elo[loser]

            expected_winner = expected_score(
                winner_rating,
                loser_rating,
            )

            change = K_FACTOR * (
                1.0 - expected_winner
            )

            overall_elo[winner] += change
            overall_elo[loser] -= change


            winner_surface_rating = (
                surface_elo[surface][winner]
            )

            loser_surface_rating = (
                surface_elo[surface][loser]
            )

            expected_surface_winner = expected_score(
                winner_surface_rating,
                loser_surface_rating,
            )

            surface_change = K_FACTOR * (
                1.0 - expected_surface_winner
            )

            surface_elo[surface][winner] += surface_change
            surface_elo[surface][loser] -= surface_change


all_players = sorted(overall_elo.keys())

rows = []

for player_key in all_players:

    rows.append(
        {
            "player_key": player_key,
            "player_name": display_names.get(
                player_key,
                player_key,
            ),
            "overall_elo": overall_elo[player_key],
            "hard_elo": surface_elo["Hard"][player_key],
            "clay_elo": surface_elo["Clay"][player_key],
            "grass_elo": surface_elo["Grass"][player_key],
        }
    )


snapshot = pd.DataFrame(rows)

snapshot = snapshot.sort_values(
    "overall_elo",
    ascending=False,
)


output_path = (
    ROOT
    / "data"
    / "current_elo_snapshot.csv"
)


snapshot.to_csv(
    output_path,
    index=False,
)


print("===== AKTUALNY SNAPSHOT ELO =====")
print("Zawodnicy:", len(snapshot))

print("\nTop 20 Overall Elo:")
print(
    snapshot[
        [
            "player_name",
            "overall_elo",
            "hard_elo",
            "clay_elo",
            "grass_elo",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\nZapisano:")
print(output_path)
