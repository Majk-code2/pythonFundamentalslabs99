import pandas as pd
import numpy as np
from collections import defaultdict

FILES = [
    "data/2022.csv",
    "data/2023.csv",
    "data/2024.csv",
    "data/2025.csv"
]

START_ELO = 1500.0
K_FACTOR = 24.0

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
    "BR": 11
}

# Wczytanie danych
data = pd.concat(
    [pd.read_csv(file) for file in FILES],
    ignore_index=True
)

# Konwersja daty
data["tourney_date"] = pd.to_datetime(
    data["tourney_date"].astype(str),
    format="%Y%m%d"
)

# Usuwamy rekordy bez potrzebnych danych
data = data.dropna(
    subset=[
        "tourney_id",
        "surface",
        "winner_name",
        "loser_name",
        "winner_rank",
        "loser_rank",
        "round"
    ]
).copy()

# Numer rundy
data["round_order"] = data["round"].map(ROUND_ORDER)

# Usuwamy nietypowe rundy, których nie umiemy poprawnie ustawić
data = data.dropna(subset=["round_order"]).copy()

data["round_order"] = data["round_order"].astype(int)

# Sortowanie chronologiczne
data = data.sort_values(
    [
        "tourney_date",
        "tourney_id",
        "round_order",
        "match_num"
    ]
).reset_index(drop=True)

# Elo ogólne
overall_elo = defaultdict(lambda: START_ELO)

# Elo osobne dla Hard, Clay i Grass
surface_elo = defaultdict(
    lambda: defaultdict(lambda: START_ELO)
)

rng = np.random.default_rng(42)

rows = []


def expected_score(rating_a, rating_b):
    return 1.0 / (
        1.0 + 10.0 ** ((rating_b - rating_a) / 400.0)
    )


# Turniej po turnieju
for (tourney_date, tourney_id), tournament in data.groupby(
    ["tourney_date", "tourney_id"],
    sort=True
):
    # Runda po rundzie
    for round_order, round_matches in tournament.groupby(
        "round_order",
        sort=True
    ):
        round_updates = []

        # Najpierw zapisujemy Elo sprzed rundy
        for _, match in round_matches.iterrows():
            winner = match["winner_name"]
            loser = match["loser_name"]
            surface = match["surface"]

            winner_elo = overall_elo[winner]
            loser_elo = overall_elo[loser]

            winner_surface_elo = surface_elo[surface][winner]
            loser_surface_elo = surface_elo[surface][loser]

            winner_is_a = rng.random() < 0.5

            if winner_is_a:
                player_a = winner
                player_b = loser

                rank_a = match["winner_rank"]
                rank_b = match["loser_rank"]

                elo_a = winner_elo
                elo_b = loser_elo

                surface_elo_a = winner_surface_elo
                surface_elo_b = loser_surface_elo

                a_won = 1

            else:
                player_a = loser
                player_b = winner

                rank_a = match["loser_rank"]
                rank_b = match["winner_rank"]

                elo_a = loser_elo
                elo_b = winner_elo

                surface_elo_a = loser_surface_elo
                surface_elo_b = winner_surface_elo

                a_won = 0

            rows.append({
                "tourney_date": tourney_date,
                "tourney_id": tourney_id,
                "tourney_name": match["tourney_name"],
                "round": match["round"],
                "surface": surface,
                "player_A": player_a,
                "player_B": player_b,
                "rank_A": rank_a,
                "rank_B": rank_b,
                "elo_A": elo_a,
                "elo_B": elo_b,
                "surface_elo_A": surface_elo_a,
                "surface_elo_B": surface_elo_b,
                "A_won": a_won
            })

            round_updates.append({
                "winner": winner,
                "loser": loser,
                "surface": surface
            })

        # Dopiero po zapisaniu wszystkich meczow rundy
        # aktualizujemy rankingi Elo
        for update in round_updates:
            winner = update["winner"]
            loser = update["loser"]
            surface = update["surface"]

            # Elo ogólne
            winner_rating = overall_elo[winner]
            loser_rating = overall_elo[loser]

            expected_winner = expected_score(
                winner_rating,
                loser_rating
            )

            elo_change = K_FACTOR * (
                1.0 - expected_winner
            )

            overall_elo[winner] += elo_change
            overall_elo[loser] -= elo_change

            # Elo na nawierzchni
            winner_surface_rating = surface_elo[surface][winner]
            loser_surface_rating = surface_elo[surface][loser]

            expected_surface_winner = expected_score(
                winner_surface_rating,
                loser_surface_rating
            )

            surface_change = K_FACTOR * (
                1.0 - expected_surface_winner
            )

            surface_elo[surface][winner] += surface_change
            surface_elo[surface][loser] -= surface_change


dataset = pd.DataFrame(rows)

dataset["rank_diff"] = (
    dataset["rank_B"] - dataset["rank_A"]
)

dataset["elo_diff"] = (
    dataset["elo_A"] - dataset["elo_B"]
)

dataset["surface_elo_diff"] = (
    dataset["surface_elo_A"]
    - dataset["surface_elo_B"]
)

dataset.to_csv(
    "data/model_dataset_elo_by_round.csv",
    index=False
)

print("===== ELO AKTUALIZOWANE PO RUNDZIE =====")
print("Liczba meczow:", len(dataset))
print("Od:", dataset["tourney_date"].min())
print("Do:", dataset["tourney_date"].max())

print("\nRundy:")
print(dataset["round"].value_counts())

print("\nPlik zapisany:")
print("data/model_dataset_elo_by_round.csv")