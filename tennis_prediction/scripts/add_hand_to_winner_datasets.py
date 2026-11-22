import numpy as np
import pandas as pd

from pathlib import Path
from player_name_utils import normalize_player_name


ROOT = Path(__file__).resolve().parents[1]


HAND_FILE = (
    ROOT
    / "data"
    / "hand_matchup_features.csv"
)


DATASETS = {
    "ATP": (
        ROOT
        / "data"
        / "model_dataset_surface_elo_return_atp.csv"
    ),

    "CHALLENGER": (
        ROOT
        / "data"
        / "model_dataset_surface_elo_return_challenger.csv"
    ),
}


hand_data = pd.read_csv(HAND_FILE)

hand_data["tourney_date"] = pd.to_datetime(
    hand_data["tourney_date"]
)


MERGE_KEYS = [
    "tourney_date",
    "tourney_id",
    "round",
    "surface",
    "competition_level",
    "pair_key",
]


HAND_COLUMNS = [
    "winner_key",
    "loser_key",
    "winner_hand",
    "loser_hand",
    "winner_matches_vs_hand",
    "loser_matches_vs_hand",
    "winner_win_rate_vs_hand",
    "loser_win_rate_vs_hand",
]


hand_lookup = (
    hand_data[
        MERGE_KEYS + HAND_COLUMNS
    ]
    .drop_duplicates(
        subset=MERGE_KEYS,
        keep="last",
    )
)


def add_hand(level, input_path):

    data = pd.read_csv(input_path)

    data["tourney_date"] = pd.to_datetime(
        data["tourney_date"]
    )

    data["player_A_key"] = (
        data["player_A"]
        .apply(normalize_player_name)
    )

    data["player_B_key"] = (
        data["player_B"]
        .apply(normalize_player_name)
    )

    data["pair_key"] = data.apply(
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

    merged = data.merge(
        hand_lookup,
        on=MERGE_KEYS,
        how="left",
        validate="many_to_one",
    )

    a_is_winner = (
        merged["player_A_key"]
        == merged["winner_key"]
    )

    merged["hand_A"] = np.where(
        a_is_winner,
        merged["winner_hand"],
        merged["loser_hand"],
    )

    merged["hand_B"] = np.where(
        a_is_winner,
        merged["loser_hand"],
        merged["winner_hand"],
    )

    merged["matches_vs_hand_A"] = np.where(
        a_is_winner,
        merged["winner_matches_vs_hand"],
        merged["loser_matches_vs_hand"],
    )

    merged["matches_vs_hand_B"] = np.where(
        a_is_winner,
        merged["loser_matches_vs_hand"],
        merged["winner_matches_vs_hand"],
    )

    merged["win_rate_vs_hand_A"] = np.where(
        a_is_winner,
        merged["winner_win_rate_vs_hand"],
        merged["loser_win_rate_vs_hand"],
    )

    merged["win_rate_vs_hand_B"] = np.where(
        a_is_winner,
        merged["loser_win_rate_vs_hand"],
        merged["winner_win_rate_vs_hand"],
    )

    # Wygładzanie do neutralnych 50%.
    # Przy małej historii surowy win rate nie dominuje.
    prior_matches = 5.0
    prior_win_rate = 0.5

    merged["smoothed_vs_hand_A"] = (
        (
            merged["win_rate_vs_hand_A"]
            * merged["matches_vs_hand_A"]
        )
        + (
            prior_win_rate
            * prior_matches
        )
    ) / (
        merged["matches_vs_hand_A"]
        + prior_matches
    )

    merged["smoothed_vs_hand_B"] = (
        (
            merged["win_rate_vs_hand_B"]
            * merged["matches_vs_hand_B"]
        )
        + (
            prior_win_rate
            * prior_matches
        )
    ) / (
        merged["matches_vs_hand_B"]
        + prior_matches
    )

    merged["hand_matchup_diff"] = (
        merged["smoothed_vs_hand_A"]
        - merged["smoothed_vs_hand_B"]
    )

    merged["is_left_A"] = (
        merged["hand_A"].eq("L").astype(int)
    )

    merged["is_left_B"] = (
        merged["hand_B"].eq("L").astype(int)
    )

    merged["left_vs_right"] = (
        merged["hand_A"].ne(merged["hand_B"])
        & merged["hand_A"].isin(["L", "R"])
        & merged["hand_B"].isin(["L", "R"])
    ).astype(int)

    merged["same_hand"] = (
        merged["hand_A"].eq(merged["hand_B"])
        & merged["hand_A"].isin(["L", "R"])
    ).astype(int)

    merged["min_matches_vs_hand"] = merged[
        [
            "matches_vs_hand_A",
            "matches_vs_hand_B",
        ]
    ].min(axis=1)

    matched = merged["winner_key"].notna().sum()

    print(
        level,
        "| dopasowanie:",
        f"{matched}/{len(merged)}",
        f"({matched / len(merged) * 100:.2f}%)",
    )

    output_path = (
        ROOT
        / "data"
        / (
            "model_dataset_surface_elo_return_hand_"
            f"{level.lower()}.csv"
        )
    )

    merged = merged.drop(
        columns=(
            HAND_COLUMNS
            + [
                "pair_key",
                "winner_key",
                "loser_key",
            ]
        ),
        errors="ignore",
    )

    merged.to_csv(
        output_path,
        index=False,
    )

    print("Zapisano:", output_path)


for level, input_path in DATASETS.items():

    add_hand(level, input_path)