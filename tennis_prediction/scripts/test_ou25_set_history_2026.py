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
        / "model_dataset_ou25_set_history_atp.csv"
    ),
    "CHALLENGER": (
        ROOT
        / "data"
        / "model_dataset_ou25_set_history_challenger.csv"
    ),
}


STRENGTH_FEATURES = [
    "abs_rank_diff",
    "abs_elo_diff",
    "abs_surface_elo_diff",
]


SET_HISTORY_FEATURES = [
    "avg_three_set_rate",
    "max_three_set_rate",
    "abs_three_set_rate_diff",
    "avg_previous_sets",
    "abs_average_sets_diff",
    "min_set_history_matches",
]


SYMMETRIC_SERVE_RETURN = [
    "abs_ace_diff",
    "abs_df_diff",
    "abs_first_serve_won_diff",
    "abs_second_serve_won_diff",
    "abs_first_return_diff",
    "abs_second_return_diff",
    "abs_return_points_diff",
    "abs_bp_converted_diff",
    "abs_matchup_diff",
]


VARIANTS = {
    "STRENGTH": STRENGTH_FEATURES,

    "STRENGTH + SET HISTORY": (
        STRENGTH_FEATURES
        + SET_HISTORY_FEATURES
    ),

    "SET HISTORY ONLY": (
        SET_HISTORY_FEATURES
    ),

    "ALL SYMMETRIC": (
        STRENGTH_FEATURES
        + SET_HISTORY_FEATURES
        + SYMMETRIC_SERVE_RETURN
    ),
}


def evaluate(
    label,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    print(f"\n--- {label} ---")

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
        "Balanced Accuracy:",
        round(
            balanced_accuracy_score(
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

    print(
        "Precision OVER:",
        round(
            precision_score(
                y_test,
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
                y_test,
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
        == y_test.to_numpy()
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

        selected_over = (
            predictions[selected] == 1
        ).sum()

        selected_under = (
            predictions[selected] == 0
        ).sum()

        print(
            f">= {int(threshold * 100)}%"
            f" | mecze: {count:4d}"
            f" | accuracy:"
            f" {selected_accuracy * 100:6.2f}%"
            f" | OVER: {selected_over:4d}"
            f" | UNDER: {selected_under:4d}"
        )


print("===== O/U 2,5 Z HISTORIA SETOWA =====")


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

    print(
        "\n"
        "==============================\n"
        f"{level}\n"
        "=============================="
    )

    print("Trening:", len(train))
    print("Test 2026:", len(test))

    print(
        "Baseline zawsze UNDER:",
        round(
            (1.0 - y_test.mean()) * 100,
            2,
        ),
        "%",
    )

    for variant_name, features in (
        VARIANTS.items()
    ):

        X_train = (
            train[features]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
        )

        X_test = (
            test[features]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
        )

        logistic_plain = LogisticRegression(
            max_iter=4000,
        )

        evaluate(
            f"LOGISTIC PLAIN | {variant_name}",
            logistic_plain,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        logistic_balanced = LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
        )

        evaluate(
            f"LOGISTIC BALANCED | {variant_name}",
            logistic_balanced,
            X_train,
            y_train,
            X_test,
            y_test,
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

        evaluate(
            f"GRADIENT | {variant_name}",
            gradient,
            X_train,
            y_train,
            X_test,
            y_test,
        )