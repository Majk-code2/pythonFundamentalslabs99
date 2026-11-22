import re
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


INVALID_MARKERS = [
    "RET",
    "W/O",
    "WO",
    "DEF",
    "ABD",
    "ABN",
    "CANCELLED",
]


SET_PATTERN = re.compile(
    r"^\d{1,2}-\d{1,2}(?:\(\d+\))?$"
)


def count_sets(score):

    if not isinstance(score, str):
        return None

    score = score.strip().upper()

    if not score:
        return None

    if any(
        marker in score
        for marker in INVALID_MARKERS
    ):
        return None

    valid_sets = [
        token
        for token in score.split()
        if SET_PATTERN.match(token)
    ]

    if len(valid_sets) in [2, 3]:
        return len(valid_sets)

    return None


def summarize(records):

    if not records:
        return {
            "set_history_matches": 0,
            "three_set_rate": 0.0,
            "straight_sets_rate": 0.0,
            "average_sets": 2.0,
        }

    frame = pd.DataFrame(list(records))

    return {
        "set_history_matches": len(frame),

        "three_set_rate": (
            frame["went_three_sets"].mean()
        ),

        "straight_sets_rate": (
            frame["straight_sets"].mean()
        ),

        "average_sets": (
            frame["sets_played"].mean()
        ),
    }


frames = []


for file_path, competition_level in FILES:

    frame = pd.read_csv(file_path)

    frame["competition_level"] = (
        competition_level
    )

    frames.append(frame)


data = pd.concat(
    frames,
    ignore_index=True,
)


data = data[
    data["best_of"].eq(3)
].copy()


data["sets_played"] = (
    data["score"].apply(count_sets)
)


data = data.dropna(
    subset=[
        "tourney_date",
        "tourney_id",
        "round",
        "surface",
        "winner_name",
        "loser_name",
        "sets_played",
    ]
).copy()


data["tourney_date"] = pd.to_datetime(
    data["tourney_date"].astype(str),
    format="%Y%m%d",
    errors="coerce",
)


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


data["went_three_sets"] = (
    data["sets_played"].eq(3).astype(int)
)


data["straight_sets"] = (
    data["sets_played"].eq(2).astype(int)
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

        updates = []

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

                    "winner_set_history_matches": (
                        winner_before[
                            "set_history_matches"
                        ]
                    ),

                    "loser_set_history_matches": (
                        loser_before[
                            "set_history_matches"
                        ]
                    ),

                    "winner_three_set_rate": (
                        winner_before[
                            "three_set_rate"
                        ]
                    ),

                    "loser_three_set_rate": (
                        loser_before[
                            "three_set_rate"
                        ]
                    ),

                    "winner_straight_sets_rate": (
                        winner_before[
                            "straight_sets_rate"
                        ]
                    ),

                    "loser_straight_sets_rate": (
                        loser_before[
                            "straight_sets_rate"
                        ]
                    ),

                    "winner_average_sets": (
                        winner_before[
                            "average_sets"
                        ]
                    ),

                    "loser_average_sets": (
                        loser_before[
                            "average_sets"
                        ]
                    ),
                }
            )

            match_record = {
                "sets_played": (
                    int(match["sets_played"])
                ),
                "went_three_sets": (
                    int(match["went_three_sets"])
                ),
                "straight_sets": (
                    int(match["straight_sets"])
                ),
            }

            updates.append(
                (winner_key, match_record)
            )

            updates.append(
                (loser_key, match_record)
            )

        for player_key, record in updates:

            history[player_key].append(record)


set_history = pd.DataFrame(rows)


output_path = (
    ROOT
    / "data"
    / "set_history_features.csv"
)


set_history.to_csv(
    output_path,
    index=False,
)


print("===== HISTORIA SETOWA =====")
print("Mecze:", len(set_history))

print(
    "Srednia historia zwyciezcy:",
    round(
        set_history[
            "winner_set_history_matches"
        ].mean(),
        2,
    ),
)

print(
    "Srednia historia przegranego:",
    round(
        set_history[
            "loser_set_history_matches"
        ].mean(),
        2,
    ),
)

print(
    "Sredni three-set rate zwyciezcy:",
    round(
        set_history[
            "winner_three_set_rate"
        ].mean() * 100,
        2,
    ),
    "%",
)

print(
    "Sredni three-set rate przegranego:",
    round(
        set_history[
            "loser_three_set_rate"
        ].mean() * 100,
        2,
    ),
    "%",
)

print("\nZapisano:")
print(output_path)