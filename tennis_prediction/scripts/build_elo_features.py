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

# Wczytanie danych
data = pd.concat(
    [pd.read_csv(file) for file in FILES],
    ignore_index=True
)

data["tourney_date"] = pd.to_datetime(
    data["tourney_date"].astype(str),
    format="%Y%m%d"
)

data = data.dropna(
    subset=[
        "tourney_id",
        "tourney_date",
        "surface",
        "winner_name",
        "loser_name",
        "winner_rank",
        "loser_rank"
    ]
).copy()

data = data.sort_values(
    ["tourney_date", "tourney_id", "match_num"]
).reset_index(drop=True)

# Elo ogólne
overall_elo = defaultdict(lambda: START_ELO)

# Elo oddzielne dla nawierzchni
surface_elo = defaultdict(
    lambda: defaultdict(lambda: START_ELO)
)

rng = np.random.default_rng(42)

rows = []

def expected_score(rating_a, rating_b):
    return 1.0 / (
        1.0 + 10.0 ** ((rating_b - rating_a) / 400.0)
    )

# Przetwarzamy turniej po turnieju
for (tourney_date, tourney_id), tournament in data.groupby(
    ["tourney_date", "tourney_id"],
    sort=True
):
    tournament_updates = []

    for _, match in tournament.iterrows():
        winner = match["winner_name"]
        loser = match["loser_name"]
        surface = match["surface"]

        winner_elo = overall_elo[winner]
        loser_elo = overall_elo[loser]

        winner_surface_elo = surface_elo[surface][winner]
        loser_surface_elo = surface_elo[surface][loser]

        # Losowo ustawiamy zwyciezce jako A albo B
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

        # Zapamiętujemy mecz, ale jeszcze nie zmieniamy Elo.
        tournament_updates.append({
            "winner": winner,
            "loser": loser,
            "surface": surface
        })

    # Aktualizacja Elo dopiero po zapisaniu cech całego turnieju
    for update in tournament_updates:
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

        change = K_FACTOR * (1.0 - expected_winner)

        overall_elo[winner] += change
        overall_elo[loser] -= change

        # Elo dla nawierzchni
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
    "data/model_dataset_with_elo.csv",
    index=False
)

print("===== DATASET Z ELO =====")
print("Liczba meczow:", len(dataset))
print("Od:", dataset["tourney_date"].min())
print("Do:", dataset["tourney_date"].max())

print("\nPrzykladowe rekordy:")
print(
    dataset[
        [
            "tourney_date",
            "surface",
            "player_A",
            "player_B",
            "rank_diff",
            "elo_diff",
            "surface_elo_diff",
            "A_won"
        ]
    ].tail(10).to_string(index=False)
)

print("\nPlik zapisany:")
print("data/model_dataset_with_elo.csv")