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
        / "model_dataset_surface_elo_return_atp.csv"
    ),
    "CHALLENGER": (
        ROOT
        / "data"
        / "model_dataset_surface_elo_return_challenger.csv"
    ),
}


BASE_ELO_FEATURES = [
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
]


MATCHUP_FEATURES = [
    "serve_return_matchup_diff",
]


VARIANTS = {
    "ELO BASE": BASE_ELO_FEATURES,

    "ELO + RETURN": (
        BASE_ELO_FEATURES
        + RETURN_FEATURES
    ),

    "ELO + MATCHUP": (
        BASE_ELO_FEATURES
        + MATCHUP_FEATURES
    ),

    "ELO + RETURN + MATCHUP": (
        BASE_ELO_FEATURES
        + RETURN_FEATURES
        + MATCHUP_FEATURES
    ),
}


def evaluate(data, label, features):

    train = data[
        data["tourney_date"].dt.year <= 2025
    ].copy()

    test = data[
        data["tourney_date"].dt.year == 2026
    ].copy()

    X_train = train[features]
    y_train = train["A_won"]

    X_test = test[features]
    y_test = test["A_won"]

    model = LogisticRegression(max_iter=4000)
    model.fit(X_train, y_train)

    probability_a = model.predict_proba(
        X_test
    )[:, 1]

    prediction = (
        probability_a >= 0.5
    ).astype(int)

    confidence = np.maximum(
        probability_a,
        1.0 - probability_a,
    )

    correct = (
        prediction == y_test.to_numpy()
    )

    print(f"\n--- {label} ---")

    print(
        "Accuracy:",
        round(
            accuracy_score(
                y_test,
                prediction,
            ) * 100,
            2,
        ),
        "%",
    )

    print(
        "Brier:",
        round(
            brier_score_loss(
                y_test,
                probability_a,
            ),
            4,
        ),
    )

    print(
        "Log Loss:",
        round(
            log_loss(
                y_test,
                probability_a,
            ),
            4,
        ),
    )

    for threshold in [0.65, 0.75, 0.85]:

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


print("===== TEST RETURN I MATCHUP NA 2026 =====")


for level, file_path in DATASETS.items():

    data = pd.read_csv(file_path)

    data["tourney_date"] = pd.to_datetime(
        data["tourney_date"]
    )

    all_features = sorted(
        {
            feature
            for feature_list in VARIANTS.values()
            for feature in feature_list
        }
    )

    data[all_features] = (
        data[all_features]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
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

    for label, features in VARIANTS.items():
        evaluate(
            data,
            label,
            features,
        )