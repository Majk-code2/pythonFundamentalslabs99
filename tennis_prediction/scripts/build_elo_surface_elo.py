import pandas as pd

from pathlib import Path
from collections import defaultdict

from player_name_utils import normalize_player_name


ROOT = Path(__file__).resolve().parents[1]

START_ELO = 1500.0
K_FACTOR = 24.0


FILES = [
    (ROOT / "data" / "2022.csv", "ATP"),
    (ROOT / "data" / "2022_challenger.csv", "CHALLENGER"),
    (ROOT / "data" / "2023.csv", "ATP"),
    (ROOT / "data" / "2023_challenger.csv", "CHALLENGER"),
    (ROOT / "data" / "2024.csv", "ATP"),
    (ROOT / "data" / "2024_challenger.csv", "CHALLENGER"),
    (ROOT / "data" / "2025.csv", "ATP"),
    (ROOT / "data" / "2025_challenger.csv", "CHALLENGER"),
    (ROOT / "data" / "2026.csv", "ATP"),
    (ROOT / "data" / "2026_challenger.csv", "CHALLENGER"),
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


frames = []

for file_path, competition_level in FILES:

    frame = pd.read_csv(file_path)

    frame["competition_level"] = competition_level

    frames.append(frame)


data = pd.concat(
    frames,
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
    data["round"]
    .map(ROUND_ORDER)
)


data = data.dropna(
    subset=["round_order"]
).copy()


data["round_order"] = (
    data["round_order"]
    .astype(int)
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


# Jedno globalne Elo dla ATP i Challengerow.
overall_elo = defaultdict(
    lambda: START_ELO
)


# Osobne Elo na Hard, Clay i Grass.
surface_elo = defaultdict(
    lambda: defaultdict(
        lambda: START_ELO
    )
)


rows = []


# Turniej po turnieju.
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

    # Runda po rundzie.
    for round_order, round_matches in tournament.groupby(
        "round_order",
        sort=True,
    ):

        round_updates = []

        # Najpierw zapisujemy Elo sprzed rundy.
        for _, match in round_matches.iterrows():

            winner = match["winner_key"]
            loser = match["loser_key"]
            surface = match["surface"]

            winner_elo = overall_elo[winner]
            loser_elo = overall_elo[loser]

            winner_surface_elo = (
                surface_elo[surface][winner]
            )

            loser_surface_elo = (
                surface_elo[surface][loser]
            )

            rows.append(
                {
                    "tourney_date": tourney_date,
                    "tourney_id": tourney_id,
                    "match_num": match["match_num"],
                    "round": match["round"],
                    "surface": surface,
                    "competition_level": (
                        match["competition_level"]
                    ),
                    "winner_name": match["winner_name"],
                    "loser_name": match["loser_name"],
                    "winner_key": winner,
                    "loser_key": loser,
                    "winner_elo_before": winner_elo,
                    "loser_elo_before": loser_elo,
                    "winner_surface_elo_before": (
                        winner_surface_elo
                    ),
                    "loser_surface_elo_before": (
                        loser_surface_elo
                    ),
                    "elo_diff_winner_loser": (
                        winner_elo - loser_elo
                    ),
                    "surface_elo_diff_winner_loser": (
                        winner_surface_elo
                        - loser_surface_elo
                    ),
                }
            )

            round_updates.append(
                (
                    winner,
                    loser,
                    surface,
                )
            )

        # Dopiero po zapisaniu calej rundy
        # aktualizujemy Elo.
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

            expected_surface_winner = (
                expected_score(
                    winner_surface_rating,
                    loser_surface_rating,
                )
            )

            surface_change = K_FACTOR * (
                1.0 - expected_surface_winner
            )

            surface_elo[surface][winner] += (
                surface_change
            )

            surface_elo[surface][loser] -= (
                surface_change
            )


elo_data = pd.DataFrame(rows)


output_path = (
    ROOT
    / "data"
    / "elo_surface_elo_history.csv"
)


elo_data.to_csv(
    output_path,
    index=False,
)


print("===== ELO I SURFACE ELO =====")
print("Liczba meczow:", len(elo_data))
print(
    "Liczba zawodnikow:",
    len(overall_elo),
)

print("\nPoziomy:")
print(
    elo_data[
        "competition_level"
    ].value_counts()
)

print("\nZakres Overall Elo:")
print(
    round(
        min(overall_elo.values()),
        2,
    ),
    "-",
    round(
        max(overall_elo.values()),
        2,
    ),
)

print("\nPrzykladowe rekordy:")
print(
    elo_data[
        [
            "tourney_date",
            "competition_level",
            "surface",
            "winner_name",
            "loser_name",
            "winner_elo_before",
            "loser_elo_before",
            "winner_surface_elo_before",
            "loser_surface_elo_before",
        ]
    ]
    .tail(10)
    .to_string(index=False)
)

print("\nZapisano:")
print(output_path)