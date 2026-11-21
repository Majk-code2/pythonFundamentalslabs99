import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

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

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Prawdopodobienstwo zwyciestwa A
prob_A = model.predict_proba(X_test)[:, 1]

prediction = (prob_A >= 0.5).astype(int)

test["prob_A"] = prob_A
test["prediction"] = prediction

# Pewnosc przewidywanego zwyciezcy
test["confidence"] = np.maximum(
    test["prob_A"],
    1 - test["prob_A"]
)

test["correct"] = (
    test["prediction"] == test["A_won"]
)

print("===== NAWIERZCHNIE - WSZYSTKIE MECZE =====")

for surface in ["Hard", "Clay", "Grass"]:

    group = test[test["surface"] == surface]

    if len(group) == 0:
        continue

    accuracy = group["correct"].mean()

    print(
        f"{surface:6} "
        f"| mecze: {len(group):4d} "
        f"| accuracy: {accuracy*100:.2f}%"
    )


print("\n===== CONFIDENCE >= 65% =====")

confident = test[test["confidence"] >= 0.65]

for surface in ["Hard", "Clay", "Grass"]:

    group = confident[
        confident["surface"] == surface
    ]

    if len(group) == 0:
        continue

    accuracy = group["correct"].mean()
    avg_confidence = group["confidence"].mean()

    print(
        f"{surface:6} "
        f"| mecze: {len(group):4d} "
        f"| accuracy: {accuracy*100:.2f}% "
        f"| avg confidence: {avg_confidence*100:.2f}%"
    )


print("\n===== CONFIDENCE >= 70% =====")

confident70 = test[test["confidence"] >= 0.70]

for surface in ["Hard", "Clay", "Grass"]:

    group = confident70[
        confident70["surface"] == surface
    ]

    if len(group) == 0:
        continue

    accuracy = group["correct"].mean()

    print(
        f"{surface:6} "
        f"| mecze: {len(group):4d} "
        f"| accuracy: {accuracy*100:.2f}%"
    )