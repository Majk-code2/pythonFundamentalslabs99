import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]


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


STRENGTH_FEATURES = [
    "abs_rank_diff",
    "abs_elo_diff",
    "abs_surface_elo_diff",
]


SERVE_FEATURES = [
    "surface_ace_diff",
    "surface_df_diff",
    "surface_first_won_diff",
    "surface_second_won_diff",
    "surface_bp_saved_diff",
]


RETURN_FEATURES = [
    "first_return_diff",
    "second_return_diff",
    "return_points_diff",
    "bp_converted_diff",
    "abs_matchup_diff",
]


FEATURE_VARIANTS = {
    "STRENGTH ONLY": (
        STRENGTH_FEATURES
    ),

    "STRENGTH + SERVE": (
        STRENGTH_FEATURES
        + SERVE_FEATURES
    ),

    "STRENGTH + SERVE + RETURN": (
        STRENGTH_FEATURES
        + SERVE_FEATURES
        + RETURN_FEATURES
    ),
}


def print_metrics(
    label,
    y_true,
    probabilities,
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    print(f"\n--- {label} ---")

    print(
        "Accuracy:",
        round(
            accuracy_score(
                y_true,
                predictions,
            ) * 100,
            2,
        ),
        "%",
    )

    print(
        "Balanced Accuracy:",
        round(
            balanced_accuracy_score(
                y_true,
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
                y_true,
                probabilities,
            ),
            4,
        ),
    )

    print(
        "Brier:",
        round(
            brier_score_loss(
                y_true,
                probabilities,
            ),
            4,
        ),
    )

    print(
        "Log Loss:",
        round(
            log_loss(
                y_true,
                probabilities,
            ),
            4,
        ),
    )

    print(
        "Precision OVER:",
        round(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            ) * 100,
            2,
        ),
        "%",
    )

    print(
        "Recall OVER:",
        round(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            ) * 100,
            2,
        ),
        "%",
    )

    confidence = np.maximum(
        probabilities,
        1.0 - probabilities,
    )

    correct = (
        predictions
        == y_true.to_numpy()
    )

    for threshold in [
        0.60,
        0.65,
        0.70,
        0.75,
    ]:

        selected = confidence >= threshold
        count = selected.sum()

        if count == 0:
            continue

        selected_accuracy = (
            correct[selected].mean()
        )

        over_selected = (
            predictions[selected] == 1
        ).sum()

        under_selected = (
            predictions[selected] == 0
        ).sum()

        print(
            f">= {int(threshold * 100)}%"
            f" | mecze: {count:4d}"
            f" | accuracy:"
            f" {selected_accuracy * 100:6.2f}%"
            f" | OVER: {over_selected:4d}"
            f" | UNDER: {under_selected:4d}"
        )


print("===== TEST OVER/UNDER 2,5 SETA =====")


for level, file_path in DATASETS.items():

    data = pd.read_csv(file_path)

    data["tourney_date"] = pd.to_datetime(
        data["tourney_date"]
    )

    train = data[
        data["tourney_date"].dt.year <= 2025
    ].copy()

    test = data[
        data["tourney_date"].dt.year == 2026
    ].copy()

    y_train = train["over_2_5_sets"]
    y_test = test["over_2_5_sets"]

    baseline_prediction = np.zeros(
        len(test),
        dtype=int,
    )

    print(
        "\n"
        "==============================\n"
        f"{level}\n"
        "=============================="
    )

    print("Trening:", len(train))
    print("Test 2026:", len(test))

    print(
        "OVER w tescie:",
        round(
            y_test.mean() * 100,
            2,
        ),
        "%",
    )

    print(
        "Baseline zawsze UNDER:",
        round(
            accuracy_score(
                y_test,
                baseline_prediction,
            ) * 100,
            2,
        ),
        "%",
    )

    for variant_name, features in (
        FEATURE_VARIANTS.items()
    ):

        train_features = (
            train[features]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
        )

        test_features = (
            test[features]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
        )

        logistic = LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
        )

        logistic.fit(
            train_features,
            y_train,
        )

        logistic_probability = (
            logistic.predict_proba(
                test_features
            )[:, 1]
        )

        print_metrics(
            f"LOGISTIC | {variant_name}",
            y_test,
            logistic_probability,
        )


        gradient = (
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42,
            )
        )

        gradient.fit(
            train_features,
            y_train,
        )

        gradient_probability = (
            gradient.predict_proba(
                test_features
            )[:, 1]
        )

        print_metrics(
            f"GRADIENT | {variant_name}",
            y_test,
            gradient_probability,
        )