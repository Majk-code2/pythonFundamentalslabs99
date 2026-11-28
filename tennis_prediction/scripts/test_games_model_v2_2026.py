import re
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)


ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "ATP": ROOT / "data/model_dataset_games_v2_atp.csv",
    "CHALLENGER": ROOT / "data/model_dataset_games_v2_challenger.csv",
}

RAW_FILES = {
    "ATP": [
        ROOT / f"data/{year}.csv"
        for year in range(2022, 2027)
    ],
    "CHALLENGER": [
        ROOT / f"data/{year}_challenger.csv"
        for year in range(2022, 2027)
    ],
}


BASE_FEATURES = [
    "rank_diff",
    "surface_win_diff",
    "surface_ace_diff",
    "surface_df_diff",
    "surface_first_won_diff",
    "surface_second_won_diff",
    "surface_bp_saved_diff",
    "elo_diff",
    "surface_elo_diff",
    "first_return_diff",
    "second_return_diff",
    "return_points_diff",
    "bp_converted_diff",
    "serve_return_matchup_diff",
]


GAMES_FEATURES = [
    "avg_total_games_pair",
    "avg_games_per_set_pair",
    "avg_tiebreak_rate_pair",
    "avg_three_set_rate_pair",
    "min_games_history_matches",
    "abs_games_history_diff",
]


SET_RE = re.compile(
    r"^(\d{1,2})-(\d{1,2})(?:\(\d+\))?$"
)


def count_games(score):

    if not isinstance(score, str):
        return np.nan

    score = score.strip().upper()

    if any(
        x in score
        for x in ["RET", "W/O", "DEF", "ABD", "ABN"]
    ):
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

    return "|||".join(
        sorted([
            str(a).strip().casefold(),
            str(b).strip().casefold(),
        ])
    )


def make_model():

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000)),
    ])


def run(level):

    data = pd.read_csv(DATASETS[level])

    data["tourney_date"] = pd.to_datetime(
        data["tourney_date"]
    )

    raw = pd.concat(
        [
            pd.read_csv(path)
            for path in RAW_FILES[level]
        ],
        ignore_index=True,
    )

    raw = raw[raw["best_of"].eq(3)].copy()

    raw["total_games"] = raw["score"].apply(
        count_games
    )

    raw["tourney_date"] = pd.to_datetime(
        raw["tourney_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )

    raw["pair_key"] = [
        pair_key(a, b)
        for a, b in zip(
            raw["winner_name"],
            raw["loser_name"],
        )
    ]

    data["pair_key"] = [
        pair_key(a, b)
        for a, b in zip(
            data["player_A"],
            data["player_B"],
        )
    ]

    keys = [
        "tourney_date",
        "tourney_id",
        "round",
        "surface",
        "pair_key",
    ]

    lookup = raw[
        keys + ["total_games"]
    ].drop_duplicates(
        subset=keys,
        keep="last",
    )

    data = data.merge(
        lookup,
        on=keys,
        how="left",
        validate="many_to_one",
    )

    data = data.dropna(
        subset=["total_games"]
    ).copy()

    train = data[
        data["tourney_date"].dt.year <= 2025
    ].copy()

    test = data[
        data["tourney_date"].dt.year == 2026
    ].copy()

    print(f"\n===== {level} =====")
    print("Trening:", len(train))
    print("Test:", len(test))

    # Wszystkie roznice sa symetryczne.
    for feature in BASE_FEATURES:

        if feature.endswith("_diff"):

            train[feature] = train[feature].abs()
            test[feature] = test[feature].abs()

    variants = {
        "V1": BASE_FEATURES,
        "V2": BASE_FEATURES + GAMES_FEATURES,
    }

    for line in [20.5, 21.5, 22.5, 23.5]:

        y_train = (
            train["total_games"] > line
        ).astype(int)

        y_test = (
            test["total_games"] > line
        ).astype(int)

        baseline_p = y_train.mean()

        print(f"\n--- OVER {line} ---")

        print(
            "Baseline Accuracy:",
            round(
                max(
                    y_test.mean(),
                    1 - y_test.mean(),
                ) * 100,
                2,
            ),
            "%",
        )

        print(
            "Baseline Brier:",
            round(
                brier_score_loss(
                    y_test,
                    np.full(len(y_test), baseline_p),
                ),
                4,
            ),
        )

        print(
            "Baseline Log Loss:",
            round(
                log_loss(
                    y_test,
                    np.full(len(y_test), baseline_p),
                ),
                4,
            ),
        )

        for name, features in variants.items():

            model = make_model()

            model.fit(
                train[features],
                y_train,
            )

            probabilities = model.predict_proba(
                test[features]
            )[:, 1]

            predictions = (
                probabilities >= 0.5
            ).astype(int)

            print(f"\n{name}")

            print(
                "Accuracy:",
                round(
                    accuracy_score(
                        y_test,
                        predictions,
                    ) * 100,
                    2,
                ),
                "%",
            )

            print(
                "ROC AUC:",
                round(
                    roc_auc_score(
                        y_test,
                        probabilities,
                    ),
                    4,
                ),
            )

            print(
                "Brier:",
                round(
                    brier_score_loss(
                        y_test,
                        probabilities,
                    ),
                    4,
                ),
            )

            print(
                "Log Loss:",
                round(
                    log_loss(
                        y_test,
                        probabilities,
                    ),
                    4,
                ),
            )


run("ATP")
run("CHALLENGER")