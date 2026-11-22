import numpy as np
import pandas as pd

from pathlib import Path
from player_name_utils import normalize_player_name


ROOT = Path(__file__).resolve().parents[1]


SET_HISTORY_FILE = (
    ROOT
    / "data"
    / "set_history_features.csv"
)


DATASETS = {
    "ATP": (
        ROOT
        / "data"
        / "model_dataset_ou25_atp.csv"
    ),

    "CHALLENGER": (
        ROOT
        / "data"
        / "model_dataset_ou25_challenger.csv"
    ),
}


set_history = pd.read_csv(
    SET_HISTORY_FILE
)


set_history["tourney_date"] = pd.to_datetime(
    set_history["tourney_date"]
)


MERGE_KEYS = [
    "tourney_date",
    "tourney_id",
    "round",
    "surface",
    "competition_level",
    "pair_key",
]


SET_COLUMNS = [
    "winner_key",
    "loser_key",
    "winner_set_history_matches",
    "loser_set_history_matches",
    "winner_three_set_rate",
    "loser_three_set_rate",
    "winner_straight_sets_rate",
    "loser_straight_sets_rate",
    "winner_average_sets",
    "loser_average_sets",
]


set_lookup = (
    set_history[
        MERGE_KEYS + SET_COLUMNS
    ]
    .drop_duplicates(
        subset=MERGE_KEYS,
        keep="last",
    )
)


def add_set_history(
    level,
    input_path,
):

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
        set_lookup,
        on=MERGE_KEYS,
        how="left",
        validate="many_to_one",
    )


    a_is_winner = (
        merged["player_A_key"]
        == merged["winner_key"]
    )


    def orient_a(
        winner_column,
        loser_column,
    ):

        return np.where(
            a_is_winner,
            merged[winner_column],
            merged[loser_column],
        )


    def orient_b(
        winner_column,
        loser_column,
    ):

        return np.where(
            a_is_winner,
            merged[loser_column],
            merged[winner_column],
        )


    merged["set_history_matches_A"] = orient_a(
        "winner_set_history_matches",
        "loser_set_history_matches",
    )

    merged["set_history_matches_B"] = orient_b(
        "winner_set_history_matches",
        "loser_set_history_matches",
    )


    merged["three_set_rate_A"] = orient_a(
        "winner_three_set_rate",
        "loser_three_set_rate",
    )

    merged["three_set_rate_B"] = orient_b(
        "winner_three_set_rate",
        "loser_three_set_rate",
    )


    merged["straight_sets_rate_A"] = orient_a(
        "winner_straight_sets_rate",
        "loser_straight_sets_rate",
    )

    merged["straight_sets_rate_B"] = orient_b(
        "winner_straight_sets_rate",
        "loser_straight_sets_rate",
    )


    merged["average_sets_A"] = orient_a(
        "winner_average_sets",
        "loser_average_sets",
    )

    merged["average_sets_B"] = orient_b(
        "winner_average_sets",
        "loser_average_sets",
    )


    # Cechy symetryczne:
    # nie zmieniaja znaczenia po zamianie A i B.

    merged["avg_three_set_rate"] = (
        merged["three_set_rate_A"]
        + merged["three_set_rate_B"]
    ) / 2.0

    merged["max_three_set_rate"] = merged[
        [
            "three_set_rate_A",
            "three_set_rate_B",
        ]
    ].max(axis=1)

    merged["abs_three_set_rate_diff"] = (
        merged["three_set_rate_A"]
        - merged["three_set_rate_B"]
    ).abs()


    merged["avg_straight_sets_rate"] = (
        merged["straight_sets_rate_A"]
        + merged["straight_sets_rate_B"]
    ) / 2.0


    merged["avg_previous_sets"] = (
        merged["average_sets_A"]
        + merged["average_sets_B"]
    ) / 2.0

    merged["abs_average_sets_diff"] = (
        merged["average_sets_A"]
        - merged["average_sets_B"]
    ).abs()


    merged["min_set_history_matches"] = merged[
        [
            "set_history_matches_A",
            "set_history_matches_B",
        ]
    ].min(axis=1)


    # Symetryczne cechy serwisu i returnu.

    merged["abs_ace_diff"] = (
        merged["surface_ace_diff"].abs()
    )

    merged["abs_df_diff"] = (
        merged["surface_df_diff"].abs()
    )

    merged["abs_first_serve_won_diff"] = (
        merged["surface_first_won_diff"].abs()
    )

    merged["abs_second_serve_won_diff"] = (
        merged["surface_second_won_diff"].abs()
    )

    merged["abs_first_return_diff"] = (
        merged["first_return_diff"].abs()
    )

    merged["abs_second_return_diff"] = (
        merged["second_return_diff"].abs()
    )

    merged["abs_bp_converted_diff"] = (
        merged["bp_converted_diff"].abs()
    )


    matched = (
        merged["winner_key"]
        .notna()
        .sum()
    )

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
            "model_dataset_ou25_set_history_"
            f"{level.lower()}.csv"
        )
    )


    columns_to_remove = (
        SET_COLUMNS
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
        output_path,
        index=False,
    )

    print("Zapisano:", output_path)


for level, input_path in DATASETS.items():

    add_set_history(
        level,
        input_path,
    )