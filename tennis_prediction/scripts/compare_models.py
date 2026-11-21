import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss
)

# Dane
data = pd.read_csv("data/model_dataset_service.csv")
data["tourney_date"] = pd.to_datetime(data["tourney_date"])

# Trening = 2022-2024
train = data[data["tourney_date"].dt.year <= 2024].copy()

# Test = 2025
test = data[data["tourney_date"].dt.year == 2025].copy()

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


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000
    ),

    "Gradient Boosting": HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42
    )
}


print("===== POROWNANIE MODELI =====")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    brier = brier_score_loss(
        y_test,
        probabilities
    )

    logloss = log_loss(
        y_test,
        probabilities
    )

    print("\n", name)

    print(
        "Accuracy:",
        round(accuracy * 100, 2),
        "%"
    )

    print(
        "Brier Score:",
        round(brier, 4)
    )

    print(
        "Log Loss:",
        round(logloss, 4)
    )


print("\n===== PUNKT ODNIESIENIA =====")
print("Ranking baseline Accuracy: 64.40%")