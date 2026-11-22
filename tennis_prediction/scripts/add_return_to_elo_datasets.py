import numpy as np
import pandas as pd

from pathlib import Path
from player_name_utils import normalize_player_name


ROOT = Path(__file__).resolve().parents[1]

RETURN_FILE = (
    ROOT
    / "data"
    / "return_features_history.csv"
)


return_data = pd.read_csv(RETURN_FILE)

return_data["tourney_date"] = pd.to_datetime(
    return_data["tourney_date"]
)


MERGE_KEYS = [
    "tourney_date",
    "tourney_id",
    "round",
    "surface",
    "competition_level",
    "pair_key",
]


RETURN_COLUMNS = [
    "winner_key",
    "loser_key",
    "winner_return_matches",
    "loser_return_matches",
    "winner_first_return",
    "loser_first_return",
    "winner_second_return",
    "loser_second_return",
    "winner_return_points",
    "loser_return_points",
    "winner_bp_converted",
    "loser_bp_converted",
]


return_lookup = (
    return_data[
        MERGE_KEYS + RETURN_COLUMNS
    ]
    .drop_duplicates(
        subset=MERGE_KEYS,
        keep="last",
    )
)


def add_return(input_name, output_name):

    model = pd.read_csv(
        ROOT / "data" / input_name
    )

    model["tourney_date"] = pd.to_datetime(
        model["tourney_date"]
    )

    model["player_A_key"] = (
        model["player_A"]
        .apply(normalize_player_name)
    )

    model["player_B_key"] = (
        model["player_B"]
        .apply(normalize_player_name)
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

    merged = model.merge(
        return_lookup,
        on=MERGE_KEYS,
        how="left",
        validate="many_to_one",
    )

    a_is_winner = (
        merged["player_A_key"]
        == merged["winner_key"]
    )

    def orient(winner_column, loser_column):
        return np.where(
            a_is_winner,
            merged[winner_column],
            merged[loser_column],
        )

    def orient_b(winner_column, loser_column):
        return np.where(
            a_is_winner,
            merged[loser_column],
            merged[winner_column],
        )

    merged["return_matches_A"] = orient(
        "winner_return_matches",
        "loser_return_matches",
    )

    merged["return_matches_B"] = orient_b(
        "winner_return_matches",
        "loser_return_matches",
    )

    merged["first_return_A"] = orient(
        "winner_first_return",
        "loser_first_return",
    )

    merged["first_return_B"] = orient_b(
        "winner_first_return",
        "loser_first_return",
    )

    merged["second_return_A"] = orient(
        "winner_second_return",
        "loser_second_return",
    )

    merged["second_return_B"] = orient_b(
        "winner_second_return",
        "loser_second_return",
    )

    merged["return_points_A"] = orient(
        "winner_return_points",
        "loser_return_points",
    )

    merged["return_points_B"] = orient_b(
        "winner_return_points",
        "loser_return_points",
    )

    merged["bp_converted_A"] = orient(
        "winner_bp_converted",
        "loser_bp_converted",
    )

    merged["bp_converted_B"] = orient_b(
        "winner_bp_converted",
        "loser_bp_converted",
    )

    merged["first_return_diff"] = (
        merged["first_return_A"]
        - merged["first_return_B"]
    )

    merged["second_return_diff"] = (
        merged["second_return_A"]
        - merged["second_return_B"]
    )

    merged["return_points_diff"] = (
        merged["return_points_A"]
        - merged["return_points_B"]
    )

    merged["bp_converted_diff"] = (
        merged["bp_converted_A"]
        - merged["bp_converted_B"]
    )


    # Dopasowanie serwis A kontra return B
    serve_a_vs_return_b = (
        merged["surface_first_won_A"]
        - merged["first_return_B"]
    )

    serve_b_vs_return_a = (
        merged["surface_first_won_B"]
        - merged["first_return_A"]
    )

    merged["serve_return_matchup_diff"] = (
        serve_a_vs_return_b
        - serve_b_vs_return_a
    )


    matched = (
        merged["winner_key"]
        .notna()
        .sum()
    )

    print(
        input_name,
        "| dopasowanie:",
        f"{matched}/{len(merged)}",
        f"({matched / len(merged) * 100:.2f}%)",
    )


    columns_to_remove = (
        RETURN_COLUMNS
        + [
            "pair_key",
            "winner_key",
            "loser_key",
        ]
    )

    merged = merged.drop(
        columns=columns_to_remove,
        errors="ignore",
    )


    merged.to_csv(
        ROOT / "data" / output_name,
        index=False,
    )


add_return(
    "model_dataset_surface_elo_atp.csv",
    "model_dataset_surface_elo_return_atp.csv",
)

add_return(
    "model_dataset_surface_elo_challenger.csv",
    "model_dataset_surface_elo_return_challenger.csv",
)