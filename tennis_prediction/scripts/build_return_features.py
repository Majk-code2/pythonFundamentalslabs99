import numpy as np
import pandas as pd

from collections import defaultdict, deque
from pathlib import Path

from player_name_utils import normalize_player_name


ROOT = Path(__file__).resolve().parents[1]


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
    "RR": 7,
    "R16": 8,
    "QF": 9,
    "SF": 10,
    "F": 11,
    "BR": 12,
}


def safe_divide(numerator, denominator):
    if (
        pd.isna(numerator)
        or pd.isna(denominator)
        or denominator <= 0
    ):
        return np.nan

    return numerator / denominator


def calculate_return_stats(server_prefix, row):
    """
    Oblicza statystyki returnujacego na podstawie
    serwisu przeciwnika.
    """

    serve_points = row.get(f"{server_prefix}_svpt")
    first_in = row.get(f"{server_prefix}_1stIn")
    first_won = row.get(f"{server_prefix}_1stWon")
    second_won = row.get(f"{server_prefix}_2ndWon")
    bp_saved = row.get(f"{server_prefix}_bpSaved")
    bp_faced = row.get(f"{server_prefix}_bpFaced")

    if (
        pd.isna(serve_points)
        or pd.isna(first_in)
        or pd.isna(first_won)
        or pd.isna(second_won)
    ):
        return {
            "first_return_won_rate": np.nan,
            "second_return_won_rate": np.nan,
            "return_points_won_rate": np.nan,
            "bp_converted_rate": np.nan,
        }

    second_serve_points = serve_points - first_in

    first_return_points_won = first_in - first_won

    second_return_points_won = (
        second_serve_points - second_won
    )

    return_points_won = (
        first_return_points_won
        + second_return_points_won
    )

    break_points_converted = (
        bp_faced - bp_saved
        if pd.notna(bp_faced) and pd.notna(bp_saved)
        else np.nan
    )

    return {
        "first_return_won_rate": safe_divide(
            first_return_points_won,
            first_in,
        ),
        "second_return_won_rate": safe_divide(
            second_return_points_won,
            second_serve_points,
        ),
        "return_points_won_rate": safe_divide(
            return_points_won,
            serve_points,
        ),
        "bp_converted_rate": safe_divide(
            break_points_converted,
            bp_faced,
        ),
    }


def summarize(records):
    if not records:
        return {
            "return_matches": 0,
            "first_return_won_rate": 0.0,
            "second_return_won_rate": 0.0,
            "return_points_won_rate": 0.0,
            "bp_converted_rate": 0.0,
        }

    frame = pd.DataFrame(list(records))

    return {
        "return_matches": len(frame),
        "first_return_won_rate": (
            frame["first_return_won_rate"].mean()
        ),
        "second_return_won_rate": (
            frame["second_return_won_rate"].mean()
        ),
        "return_points_won_rate": (
            frame["return_points_won_rate"].mean()
        ),
        "bp_converted_rate": (
            frame["bp_converted_rate"].mean()
        ),
    }


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


data["round_order"] = data["round"].map(
    ROUND_ORDER
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


history = defaultdict(
    lambda: deque(maxlen=10)
)


rows = []


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

            winner_key = match["winner_key"]
            loser_key = match["loser_key"]

            winner_before = summarize(
                history[winner_key]
            )

            loser_before = summarize(
                history[loser_key]
            )

            pair_key = "|||".join(
                sorted(
                    [
                        winner_key,
                        loser_key,
                    ]
                )
            )

            rows.append(
                {
                    "tourney_date": tourney_date,
                    "tourney_id": tourney_id,
                    "round": match["round"],
                    "surface": match["surface"],
                    "competition_level": (
                        match["competition_level"]
                    ),
                    "pair_key": pair_key,
                    "winner_key": winner_key,
                    "loser_key": loser_key,

                    "winner_return_matches": (
                        winner_before["return_matches"]
                    ),
                    "loser_return_matches": (
                        loser_before["return_matches"]
                    ),

                    "winner_first_return": (
                        winner_before[
                            "first_return_won_rate"
                        ]
                    ),
                    "loser_first_return": (
                        loser_before[
                            "first_return_won_rate"
                        ]
                    ),

                    "winner_second_return": (
                        winner_before[
                            "second_return_won_rate"
                        ]
                    ),
                    "loser_second_return": (
                        loser_before[
                            "second_return_won_rate"
                        ]
                    ),

                    "winner_return_points": (
                        winner_before[
                            "return_points_won_rate"
                        ]
                    ),
                    "loser_return_points": (
                        loser_before[
                            "return_points_won_rate"
                        ]
                    ),

                    "winner_bp_converted": (
                        winner_before[
                            "bp_converted_rate"
                        ]
                    ),
                    "loser_bp_converted": (
                        loser_before[
                            "bp_converted_rate"
                        ]
                    ),
                }
            )

            winner_return_stats = calculate_return_stats(
                "l",
                match,
            )

            loser_return_stats = calculate_return_stats(
                "w",
                match,
            )

            round_updates.append(
                (
                    winner_key,
                    winner_return_stats,
                )
            )

            round_updates.append(
                (
                    loser_key,
                    loser_return_stats,
                )
            )

        for player_key, return_stats in round_updates:
            history[player_key].append(
                return_stats
            )


return_features = pd.DataFrame(rows)


output_path = (
    ROOT
    / "data"
    / "return_features_history.csv"
)


return_features.to_csv(
    output_path,
    index=False,
)


print("===== RETURN FEATURES =====")
print("Mecze:", len(return_features))

print(
    "Srednia historia winner:",
    round(
        return_features[
            "winner_return_matches"
        ].mean(),
        2,
    ),
)

print(
    "Srednia historia loser:",
    round(
        return_features[
            "loser_return_matches"
        ].mean(),
        2,
    ),
)

print("\nOstatnie rekordy:")
print(
    return_features.tail(10).to_string(
        index=False
    )
)

print("\nZapisano:")
print(output_path)