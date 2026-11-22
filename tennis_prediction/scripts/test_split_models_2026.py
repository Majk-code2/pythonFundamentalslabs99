import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss
)


ROOT = Path(__file__).resolve().parents[1]


DATASETS = {
    "ATP TOUR": (
        ROOT
        / "data"
        / "model_dataset_surface_atp.csv"
    ),

    "CHALLENGER": (
        ROOT
        / "data"
        / "model_dataset_surface_challenger.csv"
    ),

    "POLACZONY": (
        ROOT
        / "data"
        / "model_dataset_surface.csv"
    )
}


FEATURES = [
    "rank_diff",
    "surface_win_diff",
    "surface_ace_diff",
    "surface_df_diff",
    "surface_first_won_diff",
    "surface_second_won_diff",
    "surface_bp_saved_diff"
]


def test_dataset(name, file_path):

    data = pd.read_csv(file_path)

    data["tourney_date"] = pd.to_datetime(
        data["tourney_date"]
    )

    data[FEATURES] = (
        data[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # Trening: 2022-2025
    train = data[
        data["tourney_date"].dt.year <= 2025
    ].copy()

    # Test: 2026
    test = data[
        data["tourney_date"].dt.year == 2026
    ].copy()

    if len(train) == 0 or len(test) == 0:

        print(f"\n===== {name} =====")
        print("Brak danych treningowych lub testowych.")
        return

    X_train = train[FEATURES]
    y_train = train["A_won"]

    X_test = test[FEATURES]
    y_test = test["A_won"]

    model = LogisticRegression(
        max_iter=2000
    )

    model.fit(
        X_train,
        y_train
    )

    probability_a = (
        model.predict_proba(X_test)[:, 1]
    )

    prediction = (
        probability_a >= 0.5
    ).astype(int)

    confidence = np.maximum(
        probability_a,
        1 - probability_a
    )

    correct = (
        prediction
        == y_test.to_numpy()
    )

    baseline_prediction = (
        test["rank_A"]
        < test["rank_B"]
    ).astype(int)

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_prediction
    )

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    brier = brier_score_loss(
        y_test,
        probability_a
    )

    loss = log_loss(
        y_test,
        probability_a
    )

    print(f"\n===== {name} =====")

    print(
        "Trening:",
        len(train)
    )

    print(
        "Test 2026:",
        len(test)
    )

    print(
        "Ranking baseline:",
        round(
            baseline_accuracy * 100,
            2
        ),
        "%"
    )

    print(
        "Model accuracy:",
        round(
            accuracy * 100,
            2
        ),
        "%"
    )

    print(
        "Brier Score:",
        round(brier, 4)
    )

    print(
        "Log Loss:",
        round(loss, 4)
    )

    for threshold in [
        0.65,
        0.75,
        0.85
    ]:

        selected = (
            confidence >= threshold
        )

        count = selected.sum()

        if count == 0:
            continue

        selected_accuracy = (
            correct[selected].mean()
        )

        print(
            f"Confidence >= "
            f"{int(threshold * 100)}%"
            f" | mecze: {count:4d}"
            f" | accuracy: "
            f"{selected_accuracy * 100:6.2f}%"
        )


print("===== POROWNANIE OSOBNYCH MODELI 2026 =====")


for dataset_name, dataset_path in DATASETS.items():

    test_dataset(
        dataset_name,
        dataset_path
    )