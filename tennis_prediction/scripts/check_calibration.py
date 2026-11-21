import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

data = pd.read_csv("data/model_dataset_service.csv")
data["tourney_date"] = pd.to_datetime(data["tourney_date"])

# Przeszlosc = trening
train = data[data["tourney_date"].dt.year <= 2024].copy()

# 2025 = nietkniety test
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

# Prawdopodobienstwo wygranej A
prob_A = model.predict_proba(X_test)[:, 1]

# Przewidywany zwyciezca
prediction = (prob_A >= 0.5).astype(int)

# Prawdopodobienstwo przypisane wybranemu zwyciezcy
confidence = np.maximum(prob_A, 1 - prob_A)

results = test.copy()
results["prob_A"] = prob_A
results["prediction"] = prediction
results["confidence"] = confidence
results["correct"] = prediction == y_test.to_numpy()

print("===== KALIBRACJA MODELU =====")

bins = [
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.65),
    (0.65, 0.70),
    (0.70, 0.75),
    (0.75, 0.80),
    (0.80, 0.85),
    (0.85, 0.90),
    (0.90, 1.01)
]

for low, high in bins:

    group = results[
        (results["confidence"] >= low)
        & (results["confidence"] < high)
    ]

    if len(group) < 10:
        continue

    avg_predicted = group["confidence"].mean()
    actual_accuracy = group["correct"].mean()

    difference = actual_accuracy - avg_predicted

    print(
        f"{int(low*100):02d}-{min(int(high*100),100):02d}%"
        f" | mecze: {len(group):4d}"
        f" | model: {avg_predicted*100:6.2f}%"
        f" | realnie: {actual_accuracy*100:6.2f}%"
        f" | roznica: {difference*100:+6.2f} pp"
    )

print("\n===== JAKOSC PRAWDOPODOBIENSTW =====")

print(
    "Brier Score:",
    round(brier_score_loss(y_test, prob_A), 4)
)

print(
    "Log Loss:",
    round(log_loss(y_test, prob_A), 4)
)