import pandas as pd
import numpy as np
from collections import defaultdict, deque

FILES = FILES = [
    ("data/2022.csv", "ATP"),
    ("data/2022_challenger.csv", "CHALLENGER"),
    ("data/2023.csv", "ATP"),
    ("data/2023_challenger.csv", "CHALLENGER"),
    ("data/2024.csv", "ATP"),
    ("data/2024_challenger.csv", "CHALLENGER"),
    ("data/2025.csv", "ATP"),
    ("data/2025_challenger.csv", "CHALLENGER"),
    ("data/2026.csv", "ATP"),
    ("data/2026_challenger.csv", "CHALLENGER")
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

frames = []

for file_path, competition_level in FILES:

    frame = pd.read_csv(file_path)

    frame["competition_level"] = competition_level

    frames.append(frame)


data = pd.concat(
    frames,
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

# Tylko trzy podstawowe nawierzchnie
data = data[
    data["surface"].isin(["Hard", "Clay", "Grass"])
].copy()

data["round_order"] = data["round"].map(ROUND_ORDER)
data = data.dropna(subset=["round_order"]).copy()
data["round_order"] = data["round_order"].astype(int)

data = data.sort_values(
    ["tourney_date", "tourney_id", "round_order", "match_num"]
).reset_index(drop=True)

# Osobna historia dla gracza i nawierzchni:
# history[player][surface]
history = defaultdict(
    lambda: defaultdict(
        lambda: deque(maxlen=10)
    )
)

rng = np.random.default_rng(42)
rows = []


def safe_divide(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def create_match_stats(row, prefix, won):
    service_points = row.get(f"{prefix}_svpt")
    first_in = row.get(f"{prefix}_1stIn")

    if pd.notna(service_points) and pd.notna(first_in):
        second_serve_points = service_points - first_in
    else:
        second_serve_points = np.nan

    return {
        "won": won,
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


def surface_features(player, surface):
    matches = list(history[player][surface])

    if not matches:
        return {
            "matches": 0,
            "win_rate": 0.5,
            "ace_rate": 0.0,
            "df_rate": 0.0,
            "first_won_rate": 0.0,
            "second_won_rate": 0.0,
            "bp_saved_rate": 0.0
        }

    frame = pd.DataFrame(matches)

    return {
        "matches": len(frame),
        "win_rate": frame["won"].mean(),
        "ace_rate": frame["ace_rate"].mean(),
        "df_rate": frame["df_rate"].mean(),
        "first_won_rate": frame["first_won_rate"].mean(),
        "second_won_rate": frame["second_won_rate"].mean(),
        "bp_saved_rate": frame["bp_saved_rate"].mean()
    }


# Turniej po turnieju, runda po rundzie
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
            surface = match["surface"]

            winner_stats = surface_features(winner, surface)
            loser_stats = surface_features(loser, surface)

            winner_is_a = rng.random() < 0.5

            if winner_is_a:
                player_a = winner
                player_b = loser
                rank_a = match["winner_rank"]
                rank_b = match["loser_rank"]
                stats_a = winner_stats
                stats_b = loser_stats
                a_won = 1
            else:
                player_a = loser
                player_b = winner
                rank_a = match["loser_rank"]
                rank_b = match["winner_rank"]
                stats_a = loser_stats
                stats_b = winner_stats
                a_won = 0

            rows.append({
                "tourney_date": date,
                "competition_level": match["competition_level"],
                "tourney_id": tournament_id,
                "tourney_name": match["tourney_name"],
                "round": match["round"],
                "surface": surface,
                "player_A": player_a,
                "player_B": player_b,
                "rank_A": rank_a,
                "rank_B": rank_b,
                "surface_matches_A": stats_a["matches"],
                "surface_matches_B": stats_b["matches"],
                "surface_win_rate_A": stats_a["win_rate"],
                "surface_win_rate_B": stats_b["win_rate"],
                "surface_ace_A": stats_a["ace_rate"],
                "surface_ace_B": stats_b["ace_rate"],
                "surface_df_A": stats_a["df_rate"],
                "surface_df_B": stats_b["df_rate"],
                "surface_first_won_A": stats_a["first_won_rate"],
                "surface_first_won_B": stats_b["first_won_rate"],
                "surface_second_won_A": stats_a["second_won_rate"],
                "surface_second_won_B": stats_b["second_won_rate"],
                "surface_bp_saved_A": stats_a["bp_saved_rate"],
                "surface_bp_saved_B": stats_b["bp_saved_rate"],
                "A_won": a_won
            })

            round_updates.append(
                (
                    winner,
                    surface,
                    create_match_stats(match, "w", 1)
                )
            )

            round_updates.append(
                (
                    loser,
                    surface,
                    create_match_stats(match, "l", 0)
                )
            )

        # Wyniki rundy trafiają do historii dopiero po rundzie
        for player, surface, match_stats in round_updates:
            history[player][surface].append(match_stats)


dataset = pd.DataFrame(rows)

dataset["rank_diff"] = (
    dataset["rank_B"] - dataset["rank_A"]
)

dataset["surface_win_diff"] = (
    dataset["surface_win_rate_A"]
    - dataset["surface_win_rate_B"]
)

dataset["surface_ace_diff"] = (
    dataset["surface_ace_A"]
    - dataset["surface_ace_B"]
)

dataset["surface_df_diff"] = (
    dataset["surface_df_A"]
    - dataset["surface_df_B"]
)

dataset["surface_first_won_diff"] = (
    dataset["surface_first_won_A"]
    - dataset["surface_first_won_B"]
)

dataset["surface_second_won_diff"] = (
    dataset["surface_second_won_A"]
    - dataset["surface_second_won_B"]
)

dataset["surface_bp_saved_diff"] = (
    dataset["surface_bp_saved_A"]
    - dataset["surface_bp_saved_B"]
)

feature_columns = [
    "surface_win_diff",
    "surface_ace_diff",
    "surface_df_diff",
    "surface_first_won_diff",
    "surface_second_won_diff",
    "surface_bp_saved_diff"
]

dataset[feature_columns] = (
    dataset[feature_columns].fillna(0.0)
)

# Pelny dataset
dataset.to_csv(
    "data/model_dataset_surface.csv",
    index=False
)

# Osobny dataset ATP Tour
dataset_atp = dataset[
    dataset["competition_level"] == "ATP"
].copy()

dataset_atp.to_csv(
    "data/model_dataset_surface_atp.csv",
    index=False
)

# Osobny dataset Challenger
dataset_challenger = dataset[
    dataset["competition_level"] == "CHALLENGER"
].copy()

dataset_challenger.to_csv(
    "data/model_dataset_surface_challenger.csv",
    index=False
)

print("===== DATASET WG NAWIERZCHNI =====")
print("Liczba meczow:", len(dataset))
print("Od:", dataset["tourney_date"].min())
print("Do:", dataset["tourney_date"].max())

print("\nMecze wg nawierzchni:")
print(dataset["surface"].value_counts())

print("\nSrednia historia na aktualnej nawierzchni:")
print(
    "Player A:",
    round(dataset["surface_matches_A"].mean(), 2)
)
print(
    "Player B:",
    round(dataset["surface_matches_B"].mean(), 2)
)

print("\nPlik zapisany:")
print("data/model_dataset_surface.csv")

print("\n===== PODZIAL ROZGRYWEK =====")
print("ATP Tour:", len(dataset_atp))
print("Challenger:", len(dataset_challenger))

print("\nZapisano:")
print("data/model_dataset_surface_atp.csv")
print("data/model_dataset_surface_challenger.csv")