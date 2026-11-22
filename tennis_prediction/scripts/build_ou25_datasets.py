import re
import pandas as pd

from pathlib import Path
from player_name_utils import normalize_player_name


ROOT = Path(__file__).resolve().parents[1]


RAW_FILES = [
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


MODEL_FILES = {
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


def classify_score(score):

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

    if len(valid_sets) == 2:
        return 0

    if len(valid_sets) == 3:
        return 1

    return None


raw_frames = []


for file_path, competition_level in RAW_FILES:

    frame = pd.read_csv(file_path)

    frame["competition_level"] = competition_level

    raw_frames.append(frame)


raw = pd.concat(
    raw_frames,
    ignore_index=True,
)


raw = raw[
    raw["best_of"].eq(3)
].copy()


raw["over_2_5_sets"] = (
    raw["score"].apply(classify_score)
)


raw = raw.dropna(
    subset=[
        "tourney_date",
        "tourney_id",
        "round",
        "surface",
        "winner_name",
        "loser_name",
        "over_2_5_sets",
    ]
).copy()


raw["tourney_date"] = pd.to_datetime(
    raw["tourney_date"].astype(str),
    format="%Y%m%d",
    errors="coerce",
)


raw["winner_key"] = (
    raw["winner_name"]
    .apply(normalize_player_name)
)


raw["loser_key"] = (
    raw["loser_name"]
    .apply(normalize_player_name)
)


raw["pair_key"] = raw.apply(
    lambda row: "|||".join(
        sorted(
            [
                row["winner_key"],
                row["loser_key"],
            ]
        )
    ),
    axis=1,
)


raw["over_2_5_sets"] = (
    raw["over_2_5_sets"].astype(int)
)


MERGE_KEYS = [
    "tourney_date",
    "tourney_id",
    "round",
    "surface",
    "competition_level",
    "pair_key",
]


target_lookup = raw[
    MERGE_KEYS + ["over_2_5_sets"]
].drop_duplicates(
    subset=MERGE_KEYS,
    keep="last",
)


for level, model_path in MODEL_FILES.items():

    model = pd.read_csv(model_path)

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
        target_lookup,
        on=MERGE_KEYS,
        how="inner",
        validate="many_to_one",
    )

    merged["abs_rank_diff"] = (
        merged["rank_diff"].abs()
    )

    merged["abs_elo_diff"] = (
        merged["elo_diff"].abs()
    )

    merged["abs_surface_elo_diff"] = (
        merged["surface_elo_diff"].abs()
    )

    merged["abs_return_points_diff"] = (
        merged["return_points_diff"].abs()
    )

    merged["abs_matchup_diff"] = (
        merged[
            "serve_return_matchup_diff"
        ].abs()
    )

    output_path = (
        ROOT
        / "data"
        / f"model_dataset_ou25_{level.lower()}.csv"
    )

    merged.to_csv(
        output_path,
        index=False,
    )

    print(
        level,
        "| mecze:",
        len(merged),
        "| Under:",
        merged["over_2_5_sets"].eq(0).sum(),
        "| Over:",
        merged["over_2_5_sets"].eq(1).sum(),
        "| Over %:",
        round(
            merged["over_2_5_sets"].mean() * 100,
            2,
        ),
    )

    print("Zapisano:", output_path)