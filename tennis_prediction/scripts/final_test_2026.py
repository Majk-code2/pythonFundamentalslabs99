import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss
)

# Wczytanie danych
data = pd.read_csv("data/model_dataset_service.csv")
data["tourney_date"] = pd.to_datetime(data["tourney_date"])

# Trening: wszystko przed 2026
train = data[data["tourney_date"].dt.year <= 2025].copy()

# Finalny test: tylko 2026
test = data[data["tourney_date"].dt.year == 2026].copy()

features = [
    "rank_diff",
    "ace_diff",
    "df_diff",
    "first_won_diff",
    "second_won_diff",
    "bp_saved_diff"
]

X_train = train[features]
y_train = train["A_won"]

X_test = test[features]
y_test = test["A_won"]


def evaluate_model(name, model, thresholds):

    model.fit(X_train, y_train)

    prob_a = model.predict_proba(X_test)[:, 1]
    prediction = (prob_a >= 0.5).astype(int)

    confidence = np.maximum(
        prob_a,
        1 - prob_a
    )

    correct = prediction == y_test.to_numpy()

    print(f"\n===== {name} =====")

    print(
        "Wszystkie mecze:",
        len(test)
    )

    print(
        "Accuracy:",
        round(
            accuracy_score(y_test, prediction) * 100,
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

    for threshold in thresholds:

        selected = confidence >= threshold
        count = selected.sum()

        if count == 0:
            continue

        selected_accuracy = correct[selected].mean()

        print(
            f"Confidence >= {int(threshold * 100)}%"
            f" | mecze: {count:4d}"
            f" | accuracy: {selected_accuracy * 100:6.2f}%"
        )


# Prosty baseline: wygrywa zawodnik z lepszym rankingiem
baseline_prediction = (
    test["rank_A"] < test["rank_B"]
).astype(int)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_prediction
)

print("===== FINALNY TEST 2026 =====")
print("Trening:", len(train), "meczow")
print("Test 2026:", len(test), "meczow")

print(
    "\nRanking baseline:",
    round(baseline_accuracy * 100, 2),
    "%"
)


# Logistic Regression
logistic_model = LogisticRegression(
    max_iter=2000
)

evaluate_model(
    "LOGISTIC REGRESSION",
    logistic_model,
    thresholds=[0.65]
)


# Gradient Boosting
gradient_model = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=200,
    max_leaf_nodes=15,
    l2_regularization=1.0,
    random_state=42
)

evaluate_model(
    "GRADIENT BOOSTING",
    gradient_model,
    thresholds=[0.75, 0.85]
)