import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dane
data = pd.read_csv("data/model_dataset_service.csv")
data["tourney_date"] = pd.to_datetime(data["tourney_date"])

# Trening 2022-2024
train = data[data["tourney_date"].dt.year <= 2024].copy()

# Test 2025
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

# Model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Prawdopodobienstwa zwyciestwa Player A
prob_A = model.predict_proba(X_test)[:, 1]

# Przewidywany zwyciezca
predicted = (prob_A >= 0.5).astype(int)

# Confidence = prawdopodobienstwo wybranego zwyciezcy
confidence = np.maximum(prob_A, 1 - prob_A)

results = test.copy()
results["prob_A"] = prob_A
results["prediction"] = predicted
results["confidence"] = confidence
results["correct"] = predicted == y_test.to_numpy()

print("===== CONFIDENCE ANALYSIS =====")

bins = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.01)
]

for low, high in bins:

    group = results[
        (results["confidence"] >= low)
        & (results["confidence"] < high)
    ]

    if len(group) == 0:
        continue

    accuracy = group["correct"].mean()

    print(
        f"{int(low*100)}-{min(int(high*100),100)}% "
        f"| mecze: {len(group):4d} "
        f"| trafione: {group['correct'].sum():4d} "
        f"| accuracy: {accuracy*100:.2f}%"
    )

print("\n===== MINIMALNE CONFIDENCE =====")

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

    group = results[
        results["confidence"] >= threshold
    ]

    if len(group) == 0:
        continue

    accuracy = group["correct"].mean()

    print(
        f">= {int(threshold*100)}% "
        f"| mecze: {len(group):4d} "
        f"| accuracy: {accuracy*100:.2f}%"
    )