import pandas as pd
import numpy as np

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
        / "model_dataset_surface_elo_atp.csv"
    ),
    "CHALLENGER": (
        ROOT
        / "data"
        / "model_dataset_surface_elo_challenger.csv"
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
]


VARIANTS = {
    "BASE": BASE_FEATURES,

    "BASE + OVERALL ELO": (
        BASE_FEATURES
        + ["elo_diff"]
    ),

    "BASE + SURFACE ELO": (
        BASE_FEATURES
        + ["surface_elo_diff"]
    ),

    "BASE + BOTH ELO": (
        BASE_FEATURES
        + [
            "elo_diff",
            "surface_elo_diff",
        ]
    ),
}


def evaluate_variant(
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

    X_train = train[features]
    y_train = train["A_won"]

    X_test = test[features]
    y_test = test["A_won"]

    model = LogisticRegression(
        max_iter=3000
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
        1 - probability_a,
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
            f" | accuracy: "
            f"{selected_accuracy * 100:6.2f}%"
        )


print("===== TEST ELO NA 2026 =====")


for competition_level, file_path in DATASETS.items():

    data = pd.read_csv(file_path)

    data["tourney_date"] = pd.to_datetime(
        data["tourney_date"]
    )

    all_features = list(
        {
            feature
            for features in VARIANTS.values()
            for feature in features
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
        f"\n"
        f"==============================\n"
        f"{competition_level}\n"
        f"=============================="
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
        evaluate_variant(
            data,
            label,
            features,
        )