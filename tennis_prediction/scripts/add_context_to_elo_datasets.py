import numpy as np
import pandas as pd

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


frames = []

for file_path, competition_level in FILES:
    frame = pd.read_csv(file_path)
    frame["competition_level"] = competition_level
    frames.append(frame)

raw = pd.concat(frames, ignore_index=True)

raw["tourney_date"] = pd.to_datetime(
    raw["tourney_date"].astype(str),
    format="%Y%m%d",
    errors="coerce",
)

raw["winner_key"] = raw["winner_name"].apply(
    normalize_player_name
)
raw["loser_key"] = raw["loser_name"].apply(
    normalize_player_name
)

raw["pair_key"] = raw.apply(
    lambda row: "|||".join(
        sorted([row["winner_key"], row["loser_key"]])
    ),
    axis=1,
)

raw["round_order_context"] = (
    raw["round"].map(ROUND_ORDER)
)

raw["best_of_5"] = (
    raw["best_of"].eq(5).astype(int)
)

raw["is_qualifying"] = (
    raw["round"].isin(["Q1", "Q2", "Q3"]).astype(int)
)

raw["age_diff_winner_loser"] = (
    raw["winner_age"] - raw["loser_age"]
)


def add_context(input_name, output_name):
    model = pd.read_csv(ROOT / "data" / input_name)

    model["tourney_date"] = pd.to_datetime(
        model["tourney_date"]
    )

    model["player_A_key"] = (
        model["player_A"].apply(normalize_player_name)
    )
    model["player_B_key"] = (
        model["player_B"].apply(normalize_player_name)
    )

    model["pair_key"] = model.apply(
        lambda row: "|||".join(
            sorted(
                [
                    row["player_A_key"],
                    row["player_B_key"],
                ]
            )
        ),
        axis=1,
    )

    keys = [
        "tourney_date",
        "tourney_id",
        "round",
        "surface",
        "competition_level",
        "pair_key",
    ]

    context = raw[
        keys
        + [
            "winner_key",
            "loser_key",
            "round_order_context",
            "best_of_5",
            "is_qualifying",
            "age_diff_winner_loser",
        ]
    ].drop_duplicates(
        subset=keys,
        keep="last",
    )

    merged = model.merge(
        context,
        on=keys,
        how="left",
        validate="many_to_one",
    )

    a_is_winner = (
        merged["player_A_key"]
        == merged["winner_key"]
    )

    merged["age_diff"] = np.where(
        a_is_winner,
        merged["age_diff_winner_loser"],
        -merged["age_diff_winner_loser"],
    )

    matched = merged["round_order_context"].notna().sum()

    print(
        input_name,
        "| dopasowanie:",
        f"{matched}/{len(merged)}",
        f"({matched / len(merged) * 100:.2f}%)",
    )

    merged = merged.drop(
        columns=[
            "pair_key",
            "winner_key",
            "loser_key",
            "age_diff_winner_loser",
        ],
        errors="ignore",
    )

    merged.to_csv(
        ROOT / "data" / output_name,
        index=False,
    )


add_context(
    "model_dataset_surface_elo_atp.csv",
    "model_dataset_surface_elo_context_atp.csv",
)

add_context(
    "model_dataset_surface_elo_challenger.csv",
    "model_dataset_surface_elo_context_challenger.csv",
)