import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss
)

data = pd.read_csv("data/model_dataset_surface.csv")
data["tourney_date"] = pd.to_datetime(data["tourney_date"])

# Trening: wszystko do końca 2025
train = data[data["tourney_date"].dt.year <= 2025].copy()

# Test: tylko 2026
test = data[data["tourney_date"].dt.year == 2026].copy()

features = [
    "rank_diff",
    "surface_win_diff",
    "surface_ace_diff",
    "surface_df_diff",
    "surface_first_won_diff",
    "surface_second_won_diff",
    "surface_bp_saved_diff"
]

X_train = train[features]
y_train = train["A_won"]

X_test = test[features]
y_test = test["A_won"]


def evaluate(name, model):

    model.fit(X_train, y_train)

    prob_a = model.predict_proba(X_test)[:, 1]
    predictions = (prob_a >= 0.5).astype(int)

    confidence = np.maximum(
        prob_a,
        1 - prob_a
    )

    correct = predictions == y_test.to_numpy()

    print(f"\n===== {name} =====")

    print(
        "Accuracy:",
        round(
            accuracy_score(y_test, predictions) * 100,
            2
        ),
        "%"
    )

    print(
        "Brier Score:",
        round(
            brier_score_loss(y_test, prob_a),
            4
        )
    )

    print(
        "Log Loss:",
        round(
            log_loss(y_test, prob_a),
            4
        )
    )

    for threshold in [0.65, 0.75, 0.85]:

        selected = confidence >= threshold
        count = selected.sum()

        if count == 0:
            continue

        selected_accuracy = correct[selected].mean()

        print(
            f"Confidence >= {int(threshold*100)}%"
            f" | mecze: {count:4d}"
            f" | accuracy: {selected_accuracy*100:6.2f}%"
        )

    print("\nWynik wedlug nawierzchni:")

    result = test.copy()
    result["correct"] = correct
    result["confidence"] = confidence

    for surface in ["Hard", "Clay", "Grass"]:

        group = result[
            result["surface"] == surface
        ]

        if len(group) == 0:
            continue

        surface_accuracy = group["correct"].mean()

        strong = group[
            group["confidence"] >= 0.65
        ]

        if len(strong) > 0:
            strong_accuracy = strong["correct"].mean()
        else:
            strong_accuracy = float("nan")

        print(
            f"{surface:6}"
            f" | wszystkie: {len(group):4d}"
            f" | accuracy: {surface_accuracy*100:6.2f}%"
            f" | >=65%: {len(strong):4d}"
            f" | strong accuracy: {strong_accuracy*100:6.2f}%"
        )


print("===== TEST CECH NAWIERZCHNIOWYCH 2026 =====")
print("Trening:", len(train))
print("Test:", len(test))

# Baseline rankingowy
baseline = (
    test["rank_A"] < test["rank_B"]
).astype(int)

print(
    "Ranking baseline:",
    round(
        accuracy_score(y_test, baseline) * 100,
        2
    ),
    "%"
)


evaluate(
    "LOGISTIC + SURFACE",
    LogisticRegression(max_iter=2000)
)


evaluate(
    "GRADIENT + SURFACE",
    HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42
    )
)