import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier

data = pd.read_csv("data/model_dataset_service.csv")
data["tourney_date"] = pd.to_datetime(data["tourney_date"])

# Trening: 2022-2024
train = data[data["tourney_date"].dt.year <= 2024].copy()

# Test: 2025
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

model = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=200,
    max_leaf_nodes=15,
    l2_regularization=1.0,
    random_state=42
)

model.fit(X_train, y_train)

# Prawdopodobienstwo wygranej A
prob_A = model.predict_proba(X_test)[:, 1]

prediction = (prob_A >= 0.5).astype(int)

test["prob_A"] = prob_A
test["prediction"] = prediction

test["confidence"] = np.maximum(
    test["prob_A"],
    1 - test["prob_A"]
)

test["correct"] = (
    test["prediction"] == test["A_won"]
)

print("===== GRADIENT BOOSTING - CONFIDENCE =====")

thresholds = [
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]

for threshold in thresholds:

    group = test[
        test["confidence"] >= threshold
    ]

    if len(group) == 0:
        continue

    accuracy = group["correct"].mean()
    avg_conf = group["confidence"].mean()

    print(
        f">= {int(threshold*100)}%"
        f" | mecze: {len(group):4d}"
        f" | accuracy: {accuracy*100:6.2f}%"
        f" | avg conf: {avg_conf*100:6.2f}%"
    )


print("\n===== GRADIENT >=65% WG NAWIERZCHNI =====")

confident = test[
    test["confidence"] >= 0.65
]

for surface in ["Hard", "Clay", "Grass"]:

    group = confident[
        confident["surface"] == surface
    ]

    if len(group) == 0:
        continue

    accuracy = group["correct"].mean()

    print(
        f"{surface:6}"
        f" | mecze: {len(group):4d}"
        f" | accuracy: {accuracy*100:6.2f}%"
    )