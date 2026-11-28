import re
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "ATP": ROOT / "data/model_dataset_surface_elo_return_atp.csv",
    "CHALLENGER": ROOT / "data/model_dataset_surface_elo_return_challenger.csv",
}

RAW_FILES = {
    "ATP": ROOT / "data/2026.csv",
    "CHALLENGER": ROOT / "data/2026_challenger.csv",
}

BASE = [
    "rank_diff",
    "surface_win_diff",
    "surface_ace_diff",
    "surface_df_diff",
    "surface_first_won_diff",
    "surface_second_won_diff",
    "surface_bp_saved_diff",
    "elo_diff",
    "surface_elo_diff",
]

FEATURES = {
    "ATP": BASE + ["serve_return_matchup_diff"],
    "CHALLENGER": BASE + [
        "first_return_diff",
        "second_return_diff",
        "return_points_diff",
        "bp_converted_diff",
        "serve_return_matchup_diff",
    ],
}

BINS = [0.50, 0.55, 0.60, 0.70, 0.80, 1.00001]
LABELS = ["50-55%", "55-60%", "60-70%", "70-80%", "80-100%"]

SET_RE = re.compile(r"^(\d{1,2})-(\d{1,2})(?:\(\d+\))?$")


def total_games(score):
    if not isinstance(score, str):
        return np.nan

    score = score.strip().upper()

    if any(x in score for x in ["RET", "W/O", "DEF", "ABD", "ABN"]):
        return np.nan

    tokens = score.split()

    if len(tokens) not in (2, 3):
        return np.nan

    total = 0
    winner_sets = 0
    loser_sets = 0

    for token in tokens:
        match = SET_RE.fullmatch(token)

        if match is None:
            return np.nan

        a, b = map(int, match.groups())
        total += a + b

        winner_sets += int(a > b)
        loser_sets += int(b > a)

    if winner_sets != 2 or loser_sets not in (0, 1):
        return np.nan

    return total


def pair_key(a, b):
    return "|||".join(sorted([
        str(a).strip().casefold(),
        str(b).strip().casefold(),
    ]))


def run(level):
    data = pd.read_csv(DATASETS[level])
    data["tourney_date"] = pd.to_datetime(data["tourney_date"])

    train = data[data["tourney_date"].dt.year <= 2025].copy()
    test = data[data["tourney_date"].dt.year == 2026].copy()

    features = FEATURES[level]

    X_train = (
        train[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    X_test = (
        test[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000)
    )

    model.fit(X_train, train["A_won"])

    p_a = model.predict_proba(X_test)[:, 1]
    test["confidence"] = np.maximum(p_a, 1 - p_a)

    raw = pd.read_csv(RAW_FILES[level])
    raw = raw[raw["best_of"].eq(3)].copy()

    raw["total_games"] = raw["score"].apply(total_games)

    raw["tourney_date"] = pd.to_datetime(
        raw["tourney_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )

    raw["pair_key"] = [
        pair_key(a, b)
        for a, b in zip(raw["winner_name"], raw["loser_name"])
    ]

    test["pair_key"] = [
        pair_key(a, b)
        for a, b in zip(test["player_A"], test["player_B"])
    ]

    keys = [
        "tourney_date",
        "tourney_id",
        "round",
        "surface",
        "pair_key",
    ]

    lookup = raw[keys + ["total_games"]].drop_duplicates(
        subset=keys,
        keep="last",
    )

    merged = test.merge(
        lookup,
        on=keys,
        how="left",
        validate="many_to_one",
    )

    merged = merged.dropna(subset=["total_games"]).copy()

    merged["confidence_group"] = pd.cut(
        merged["confidence"],
        bins=BINS,
        labels=LABELS,
        include_lowest=True,
        right=False,
    )

    for line in [20.5, 21.5, 22.5, 23.5]:
        merged[f"over_{line}"] = (
            merged["total_games"] > line
        ).astype(int)

    result = merged.groupby(
        "confidence_group",
        observed=True,
    ).agg(
        matches=("total_games", "size"),
        avg_games=("total_games", "mean"),
        median_games=("total_games", "median"),
        over_20_5=("over_20.5", "mean"),
        over_21_5=("over_21.5", "mean"),
        over_22_5=("over_22.5", "mean"),
        over_23_5=("over_23.5", "mean"),
    )

    for col in [
        "over_20_5",
        "over_21_5",
        "over_22_5",
        "over_23_5",
    ]:
        result[col] = (result[col] * 100).round(2)

    print(f"\n===== {level} =====")
    print("Trening:", len(train))
    print("Test 2026:", len(test))
    print("Mecze z poprawnym wynikiem:", len(merged))
    print(result.round(2).to_string())

    return merged


atp = run("ATP")
challenger = run("CHALLENGER")

print("\nEksperyment zakonczony.")