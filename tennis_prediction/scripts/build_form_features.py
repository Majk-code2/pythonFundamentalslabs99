import pandas as pd
import numpy as np
from collections import defaultdict, deque

files = [
    "data/2022.csv",
    "data/2023.csv",
    "data/2024.csv",
    "data/2025.csv"
]

# Wczytanie wszystkich meczow
data = pd.concat(
    [pd.read_csv(file) for file in files],
    ignore_index=True
)

# Przygotowanie daty
data["tourney_date"] = pd.to_datetime(
    data["tourney_date"].astype(str),
    format="%Y%m%d"
)

# Usuwamy rekordy bez rankingu lub nazwiska
data = data.dropna(
    subset=[
        "winner_name",
        "loser_name",
        "winner_rank",
        "loser_rank"
    ]
).copy()

# Ustalamy stala kolejnosc
data = data.sort_values(
    ["tourney_date", "tourney_name", "round"]
).reset_index(drop=True)

# Historia wynikow kazdego zawodnika:
# 1 = zwyciestwo, 0 = porazka
player_history = defaultdict(lambda: deque(maxlen=5))

# Generator losowy, zawsze da ten sam wynik
rng = np.random.default_rng(42)

rows = []

# Przetwarzamy mecze dzien po dniu
for match_date, daily_matches in data.groupby(
    "tourney_date",
    sort=True
):
    daily_updates = []

    for _, match in daily_matches.iterrows():
        winner = match["winner_name"]
        loser = match["loser_name"]

        winner_history = list(player_history[winner])
        loser_history = list(player_history[loser])

        winner_form5 = (
            sum(winner_history) / len(winner_history)
            if winner_history
            else 0.5
        )

        loser_form5 = (
            sum(loser_history) / len(loser_history)
            if loser_history
            else 0.5
        )

        # Losujemy, czy zwyciezca bedzie Player A czy Player B
        winner_is_a = rng.random() < 0.5

        if winner_is_a:
            player_a = winner
            player_b = loser

            rank_a = match["winner_rank"]
            rank_b = match["loser_rank"]

            form5_a = winner_form5
            form5_b = loser_form5

            matches_before_a = len(winner_history)
            matches_before_b = len(loser_history)

            a_won = 1

        else:
            player_a = loser
            player_b = winner

            rank_a = match["loser_rank"]
            rank_b = match["winner_rank"]

            form5_a = loser_form5
            form5_b = winner_form5

            matches_before_a = len(loser_history)
            matches_before_b = len(winner_history)

            a_won = 0

        rows.append({
            "tourney_date": match_date,
            "tourney_name": match["tourney_name"],
            "surface": match["surface"],
            "player_A": player_a,
            "player_B": player_b,
            "rank_A": rank_a,
            "rank_B": rank_b,
            "form5_A": form5_a,
            "form5_B": form5_b,
            "matches_before_A": matches_before_a,
            "matches_before_B": matches_before_b,
            "A_won": a_won
        })

        # Aktualizacje wykonamy dopiero po przetworzeniu calego dnia
        daily_updates.append((winner, 1))
        daily_updates.append((loser, 0))

    # Dopiero teraz dopisujemy wyniki tego dnia do historii
    for player, result in daily_updates:
        player_history[player].append(result)

dataset = pd.DataFrame(rows)

dataset["rank_diff"] = dataset["rank_B"] - dataset["rank_A"]
dataset["form5_diff"] = dataset["form5_A"] - dataset["form5_B"]

dataset.to_csv(
    "data/model_dataset_with_form.csv",
    index=False
)

print("===== DATASET Z FORMA =====")
print("Liczba meczow:", len(dataset))
print("Od:", dataset["tourney_date"].min())
print("Do:", dataset["tourney_date"].max())

print("\nPierwsze rekordy:")
print(dataset.head(10).to_string(index=False))

print("\nRozklad wyniku:")
print(dataset["A_won"].value_counts())

print("\nSrednia forma Player A:")
print(round(dataset["form5_A"].mean(), 3))

print("\nPlik zapisany:")
print("data/model_dataset_with_form.csv")