import numpy as np
import pandas as pd

from pathlib import Path
from player_name_utils import normalize_player_name


ROOT = Path(__file__).resolve().parents[1]

HISTORY_FILE = (
    ROOT / "data" / "games_history_features.csv"
)

DATASETS = {
    "ATP": (
        ROOT / "data"
        / "model_dataset_surface_elo_return_atp.csv"
    ),
    "CHALLENGER": (
        ROOT / "data"
        / "model_dataset_surface_elo_return_challenger.csv"
    ),
}


MERGE_KEYS = [
    "tourney_date",
    "tourney_id",
    "match_num",
    "round",
    "surface",
    "competition_level",
    "pair_key",
]


HISTORY_METRICS = [
    "games_history_matches",
    "avg_total_games",
    "avg_games_per_set",
    "tiebreak_rate",
    "three_set_rate",
]


history = pd.read_csv(HISTORY_FILE)

history["tourney_date"] = pd.to_datetime(
    history["tourney_date"]
)

# Zachowujemy identyfikatory winner/loser,
# aby poprawnie przypisac historie do A/B.
# Plik historii nie zawiera winner_key/loser_key,
# dlatego odtwarzamy orientacje z surowych CSV.

raw_files = [
    (ROOT / "data" / f"{year}.csv", "ATP")
    for year in range(2022, 2027)
] + [
    (ROOT / "data" / f"{year}_challenger.csv", "CHALLENGER")
    for year in range(2022, 2027)
]

raw_frames = []

for path, level in raw_files:
    raw = pd.read_csv(path)
    raw["competition_level"] = level
    raw_frames.append(raw)

raw = pd.concat(raw_frames, ignore_index=True)

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

raw["pair_key"] = [
    "|||".join(sorted([a, b]))
    for a, b in zip(raw["winner_key"], raw["loser_key"])
]

raw_keys = raw[
    MERGE_KEYS + ["winner_key", "loser_key"]
].drop_duplicates(subset=MERGE_KEYS)

history = history.merge(
    raw_keys,
    on=MERGE_KEYS,
    how="left",
    validate="many_to_one",
)

# Nie dopuszczamy do wieloznacznych kluczy historii.
if history.duplicated(MERGE_KEYS).any():
    raise ValueError(
        "Duplikaty kluczy w games_history_features.csv"
    )


def build_dataset(level, input_path):

    data = pd.read_csv(input_path)

    data["tourney_date"] = pd.to_datetime(
        data["tourney_date"]
    )

    data["player_A_key"] = data["player_A"].apply(
        normalize_player_name
    )

    data["player_B_key"] = data["player_B"].apply(
        normalize_player_name
    )

    data["pair_key"] = [
        "|||".join(sorted([a, b]))
        for a, b in zip(
            data["player_A_key"],
            data["player_B_key"],
        )
    ]

       # Uzupelniamy match_num z danych zrodlowych.
    # Tylko jednoznaczne dopasowania.

    raw_match_lookup = raw[
        MERGE_KEYS
    ].drop_duplicates()

    keys_without_match_num = [
        key
        for key in MERGE_KEYS
        if key != "match_num"
    ]

    # Sprawdzamy, czy jeden zestaw kluczy
    # nie odpowiada kilku match_num.
    ambiguous = raw_match_lookup.duplicated(
        keys_without_match_num,
        keep=False
    )

    if ambiguous.any():

        print(
            level,
            "| niejednoznaczne mecze:",
            ambiguous.sum()
        )

        raw_match_lookup = raw_match_lookup[
            ~ambiguous
        ].copy()

    data = data.merge(
        raw_match_lookup,
        on=keys_without_match_num,
        how="left",
        validate="many_to_one",
    )

    print(
        level,
        "| uzupelnione match_num:",
        data["match_num"].notna().sum(),
        "/",
        len(data)
    )
    merged = data.merge(
        history,
        on=MERGE_KEYS,
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    matched = merged["_merge"].eq("both")

    print(
        level,
        "| dopasowanie:",
        f"{matched.sum()}/{len(merged)}",
        f"({matched.mean() * 100:.2f}%)",
    )

    # Kontrola orientacji.
    a_is_winner = (
        merged["player_A_key"] == merged["winner_key"]
    )

    a_is_loser = (
        merged["player_A_key"] == merged["loser_key"]
    )

    valid_orientation = (
        ~matched | (a_is_winner | a_is_loser)
    )

    if not valid_orientation.all():
        raise ValueError(
            f"{level}: niezgodnosc player_A z winner/loser"
        )

    for metric in HISTORY_METRICS:

        winner_col = f"winner_{metric}"
        loser_col = f"loser_{metric}"

        merged[f"{metric}_A"] = np.where(
            a_is_winner,
            merged[winner_col],
            merged[loser_col],
        )

        merged[f"{metric}_B"] = np.where(
            a_is_winner,
            merged[loser_col],
            merged[winner_col],
        )

    # Symetryczne cechy pary.
    merged["avg_total_games_pair"] = (
        merged["avg_total_games_A"]
        + merged["avg_total_games_B"]
    ) / 2

    merged["avg_games_per_set_pair"] = (
        merged["avg_games_per_set_A"]
        + merged["avg_games_per_set_B"]
    ) / 2

    merged["avg_tiebreak_rate_pair"] = (
        merged["tiebreak_rate_A"]
        + merged["tiebreak_rate_B"]
    ) / 2

    merged["avg_three_set_rate_pair"] = (
        merged["three_set_rate_A"]
        + merged["three_set_rate_B"]
    ) / 2

    merged["abs_games_history_diff"] = (
        merged["avg_total_games_A"]
        - merged["avg_total_games_B"]
    ).abs()

    merged["min_games_history_matches"] = merged[
        [
            "games_history_matches_A",
            "games_history_matches_B",
        ]
    ].min(axis=1)

    output = (
        ROOT / "data"
        / f"model_dataset_games_v2_{level.lower()}.csv"
    )

    # Usuwamy pomocnicze kolumny z merge.
    merged = merged.drop(
        columns=[
            "_merge",
            "winner_key",
            "loser_key",
            "player_A_key",
            "player_B_key",
            "pair_key",
        ],
        errors="ignore",
    )

    merged.to_csv(output, index=False)

    print("Zapisano:", output)
    print("Wiersze:", len(merged))


for level, path in DATASETS.items():
    build_dataset(level, path)