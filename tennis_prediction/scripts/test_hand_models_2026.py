import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
)


ROOT = Path(__file__).resolve().parents[1]


DATASETS = {
    "ATP": (
        ROOT
        / "data"
        / "model_dataset_surface_elo_return_hand_atp.csv"
    ),

    "CHALLENGER": (
        ROOT
        / "data"
        / "model_dataset_surface_elo_return_hand_challenger.csv"
    ),
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
]


RETURN_FEATURES = [
    "first_return_diff",
    "second_return_diff",
    "return_points_diff",
    "bp_converted_diff",
    "serve_return_matchup_diff",
]


HAND_FLAGS = [
    "is_left_A",
    "is_left_B",
    "left_vs_right",
    "same_hand",
]


HAND_HISTORY = [
    "hand_matchup_diff",
    "min_matches_vs_hand",
]


VARIANTS_BY_LEVEL = {
    "ATP": {
        "CURRENT BEST": (
            BASE_FEATURES
            + ["serve_return_matchup_diff"]
        ),

        "CURRENT + HAND FLAGS": (
            BASE_FEATURES
            + ["serve_return_matchup_diff"]
            + HAND_FLAGS
        ),

        "CURRENT + HAND HISTORY": (
            BASE_FEATURES
            + ["serve_return_matchup_diff"]
            + HAND_HISTORY
        ),

        "CURRENT + ALL HAND": (
            BASE_FEATURES
            + ["serve_return_matchup_diff"]
            + HAND_FLAGS
            + HAND_HISTORY
        ),
    },

    "CHALLENGER": {
        "CURRENT BEST": (
            BASE_FEATURES
            + RETURN_FEATURES
        ),

        "CURRENT + HAND FLAGS": (
            BASE_FEATURES
            + RETURN_FEATURES
            + HAND_FLAGS
        ),

        "CURRENT + HAND HISTORY": (
            BASE_FEATURES
            + RETURN_FEATURES
            + HAND_HISTORY
        ),

        "CURRENT + ALL HAND": (
            BASE_FEATURES
            + RETURN_FEATURES
            + HAND_FLAGS
            + HAND_HISTORY
        ),
    },
}


def evaluate(
    data,
    label,
    features,
):

    train = data[
        data["tourney_date"].dt.year <= 2025
    ].copy()

    test = data[
        data["tourney_date"].dt.year == 2026
    ].copy()

    X_train = (
        train[features]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    y_train = train["A_won"]

    X_test = (
        test[features]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    y_test = test["A_won"]

    model = LogisticRegression(
        max_iter=5000
    )

    model.fit(
        X_train,
        y_train,
    )

    probability_a = (
        model.predict_proba(X_test)[:, 1]
    )

    prediction = (
        probability_a >= 0.5
    ).astype(int)

    confidence = np.maximum(
        probability_a,
        1.0 - probability_a,
    )

    correct = (
        prediction
        == y_test.to_numpy()
    )

    accuracy = accuracy_score(
        y_test,
        prediction,
    )

    brier = brier_score_loss(
        y_test,
        probability_a,
    )

    loss = log_loss(
        y_test,
        probability_a,
    )

    print(f"\n--- {label} ---")

    print(
        "Accuracy:",
        round(accuracy * 100, 2),
        "%",
    )

    print(
        "Brier:",
        round(brier, 4),
    )

    print(
        "Log Loss:",
        round(loss, 4),
    )

    for threshold in [
        0.65,
        0.75,
        0.85,
    ]:

        selected = confidence >= threshold
        count = selected.sum()

        if count == 0:
            continue

        selected_accuracy = (
            correct[selected].mean()
        )

        print(
            f">= {int(threshold * 100)}%"
            f" | mecze: {count:4d}"
            f" | accuracy:"
            f" {selected_accuracy * 100:6.2f}%"
        )


print("===== TEST CECH REKI L/R NA 2026 =====")


for level, file_path in DATASETS.items():

    data = pd.read_csv(file_path)

    data["tourney_date"] = pd.to_datetime(
        data["tourney_date"]
    )

    print(
        "\n"
        "==============================\n"
        f"{level}\n"
        "=============================="
    )

    print(
        "Trening:",
        (
            data["tourney_date"].dt.year
            <= 2025
        ).sum(),
    )

    print(
        "Test 2026:",
        (
            data["tourney_date"].dt.year
            == 2026
        ).sum(),
    )

    print(
        "Braki hand_matchup_diff:",
        data["hand_matchup_diff"]
        .isna()
        .sum(),
    )

    for label, features in (
        VARIANTS_BY_LEVEL[level].items()
    ):

        evaluate(
            data,
            label,
            features,
        )