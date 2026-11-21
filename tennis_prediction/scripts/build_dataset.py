import pandas as pd
import numpy as np

files = [
    "data/2022.csv",
    "data/2023.csv",
    "data/2024.csv",
    "data/2025.csv"
]

# Wczytanie danych
data = pd.concat(
    [pd.read_csv(file) for file in files],
    ignore_index=True
)

# Data
data["tourney_date"] = pd.to_datetime(
    data["tourney_date"].astype(str),
    format="%Y%m%d"
)

data = data.sort_values("tourney_date").reset_index(drop=True)

# Na początek usuwamy mecze bez rankingu
data = data.dropna(subset=["winner_rank", "loser_rank"]).copy()

# Losowo ustawiamy zwyciezce jako A lub B
rng = np.random.default_rng(42)
winner_is_a = rng.random(len(data)) < 0.5

data["player_A"] = np.where(
    winner_is_a,
    data["winner_name"],
    data["loser_name"]
)

data["player_B"] = np.where(
    winner_is_a,
    data["loser_name"],
    data["winner_name"]
)

data["rank_A"] = np.where(
    winner_is_a,
    data["winner_rank"],
    data["loser_rank"]
)

data["rank_B"] = np.where(
    winner_is_a,
    data["loser_rank"],
    data["winner_rank"]
)

# 1 = Player A wygral
# 0 = Player B wygral
data["A_won"] = winner_is_a.astype(int)

dataset = data[
    [
        "tourney_date",
        "tourney_name",
        "surface",
        "player_A",
        "player_B",
        "rank_A",
        "rank_B",
        "A_won"
    ]
].copy()

dataset.to_csv("data/model_dataset.csv", index=False)

print("===== MODEL DATASET =====")
print("Liczba meczow:", len(dataset))
print()
print(dataset.head(10).to_string(index=False))

print("\nRozklad wyniku:")
print(dataset["A_won"].value_counts())