import pandas as pd
import numpy as np
from collections import defaultdict, deque

FILES = [
    "data/2022.csv",
    "data/2023.csv",
    "data/2024.csv",
    "data/2025.csv",
    "data/2026.csv"
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
    "BR": 11
}

# Wczytanie danych
data = pd.concat(
    [pd.read_csv(file) for file in FILES],
    ignore_index=True
)

data["tourney_date"] = pd.to_datetime(
    data["tourney_date"].astype(str),
    format="%Y%m%d"
)

required = [
    "tourney_id",
    "tourney_date",
    "surface",
    "round",
    "winner_name",
    "loser_name",
    "winner_rank",
    "loser_rank"
]

data = data.dropna(subset=required).copy()

data["round_order"] = data["round"].map(ROUND_ORDER)
data = data.dropna(subset=["round_order"]).copy()
data["round_order"] = data["round_order"].astype(int)

data = data.sort_values(
    ["tourney_date", "tourney_id", "round_order", "match_num"]
).reset_index(drop=True)

# Ostatnie 10 wystepow każdego zawodnika
history = defaultdict(lambda: deque(maxlen=10))

rng = np.random.default_rng(42)
rows = []


def safe_divide(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def create_match_stats(row, prefix):
    service_points = row.get(f"{prefix}_svpt")
    first_in = row.get(f"{prefix}_1stIn")

    second_serve_points = (
        service_points - first_in
        if pd.notna(service_points) and pd.notna(first_in)
        else np.nan
    )

    return {
        "ace_rate": safe_divide(
            row.get(f"{prefix}_ace"),
            service_points
        ),
        "df_rate": safe_divide(
            row.get(f"{prefix}_df"),
            service_points
        ),
        "first_won_rate": safe_divide(
            row.get(f"{prefix}_1stWon"),
            first_in
        ),
        "second_won_rate": safe_divide(
            row.get(f"{prefix}_2ndWon"),
            second_serve_points
        ),
        "bp_saved_rate": safe_divide(
            row.get(f"{prefix}_bpSaved"),
            row.get(f"{prefix}_bpFaced")
        )
    }


def historical_features(player):
    matches = list(history[player])

    if not matches:
        return {
            "matches": 0,
            "ace_rate": 0.0,
            "df_rate": 0.0,
            "first_won_rate": 0.0,
            "second_won_rate": 0.0,
            "bp_saved_rate": 0.0
        }

    frame = pd.DataFrame(matches)

    return {
        "matches": len(frame),
        "ace_rate": frame["ace_rate"].mean(),
        "df_rate": frame["df_rate"].mean(),
        "first_won_rate": frame["first_won_rate"].mean(),
        "second_won_rate": frame["second_won_rate"].mean(),
        "bp_saved_rate": frame["bp_saved_rate"].mean()
    }


# Turniej po turnieju oraz runda po rundzie
for (date, tournament_id), tournament in data.groupby(
    ["tourney_date", "tourney_id"],
    sort=True
):
    for round_number, round_matches in tournament.groupby(
        "round_order",
        sort=True
    ):
        round_updates = []

        for _, match in round_matches.iterrows():
            winner = match["winner_name"]
            loser = match["loser_name"]

            winner_history = historical_features(winner)
            loser_history = historical_features(loser)

            winner_is_a = rng.random() < 0.5

            if winner_is_a:
                player_a = winner
                player_b = loser
                rank_a = match["winner_rank"]
                rank_b = match["loser_rank"]
                stats_a = winner_history
                stats_b = loser_history
                a_won = 1
            else:
                player_a = loser
                player_b = winner
                rank_a = match["loser_rank"]
                rank_b = match["winner_rank"]
                stats_a = loser_history
                stats_b = winner_history
                a_won = 0

            rows.append({
                "tourney_date": date,
                "tourney_id": tournament_id,
                "tourney_name": match["tourney_name"],
                "round": match["round"],
                "surface": match["surface"],
                "player_A": player_a,
                "player_B": player_b,
                "rank_A": rank_a,
                "rank_B": rank_b,
                "matches_A": stats_a["matches"],
                "matches_B": stats_b["matches"],
                "ace_rate_A": stats_a["ace_rate"],
                "ace_rate_B": stats_b["ace_rate"],
                "df_rate_A": stats_a["df_rate"],
                "df_rate_B": stats_b["df_rate"],
                "first_won_A": stats_a["first_won_rate"],
                "first_won_B": stats_b["first_won_rate"],
                "second_won_A": stats_a["second_won_rate"],
                "second_won_B": stats_b["second_won_rate"],
                "bp_saved_A": stats_a["bp_saved_rate"],
                "bp_saved_B": stats_b["bp_saved_rate"],
                "A_won": a_won
            })

            winner_stats = create_match_stats(match, "w")
            loser_stats = create_match_stats(match, "l")

            round_updates.append(
                (winner, winner_stats)
            )
            round_updates.append(
                (loser, loser_stats)
            )

        # Statystyki z rundy dodajemy dopiero po jej zakończeniu
        for player, match_stats in round_updates:
            history[player].append(match_stats)


dataset = pd.DataFrame(rows)

dataset["rank_diff"] = (
    dataset["rank_B"] - dataset["rank_A"]
)

dataset["ace_diff"] = (
    dataset["ace_rate_A"] - dataset["ace_rate_B"]
)

dataset["df_diff"] = (
    dataset["df_rate_A"] - dataset["df_rate_B"]
)

dataset["first_won_diff"] = (
    dataset["first_won_A"] - dataset["first_won_B"]
)

dataset["second_won_diff"] = (
    dataset["second_won_A"] - dataset["second_won_B"]
)

dataset["bp_saved_diff"] = (
    dataset["bp_saved_A"] - dataset["bp_saved_B"]
)

# Średnie mogą mieć braki, gdy wcześniejsze mecze
# nie zawierały pełnych statystyk
feature_columns = [
    "ace_diff",
    "df_diff",
    "first_won_diff",
    "second_won_diff",
    "bp_saved_diff"
]

dataset[feature_columns] = dataset[feature_columns].fillna(0.0)

dataset.to_csv(
    "data/model_dataset_service.csv",
    index=False
)

print("===== DATASET ZE STATYSTYKAMI SERWISU =====")
print("Liczba meczow:", len(dataset))
print("Od:", dataset["tourney_date"].min())
print("Do:", dataset["tourney_date"].max())

print("\nSrednia liczba wcześniejszych meczow:")
print("Player A:", round(dataset["matches_A"].mean(), 2))
print("Player B:", round(dataset["matches_B"].mean(), 2))

print("\nPlik zapisany:")
print("data/model_dataset_service.csv")