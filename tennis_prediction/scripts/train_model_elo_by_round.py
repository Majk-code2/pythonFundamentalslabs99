import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss
)

data = pd.read_csv("data/model_dataset_elo_by_round.csv")
data["tourney_date"] = pd.to_datetime(data["tourney_date"])

# Trening tylko na przeszlosci
train = data[data["tourney_date"].dt.year <= 2024].copy()

# Test na 2025
test = data[data["tourney_date"].dt.year == 2025].copy()

features = [
    "rank_diff",
    "elo_diff",
    "surface_elo_diff"
]

X_train = train[features]
y_train = train["A_won"]

X_test = test[features]
y_test = test["A_won"]

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
logloss = log_loss(y_test, probabilities)
brier = brier_score_loss(y_test, probabilities)

print("===== RANKING + ELO + SURFACE ELO =====")
print("Trening:", len(train))
print("Test:", len(test))

print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("Log Loss:", round(logloss, 4))
print("Brier Score:", round(brier, 4))

print("\nPOROWNANIE")
print("Ranking baseline:       64.40%")
print("Ranking ML:             64.04%")
print("Ranking + forma5:       62.83%")
print("Ranking + Elo + Surface:", round(accuracy * 100, 2), "%")

print("\nWAGI MODELU")
for name, coef in zip(features, model.coef_[0]):
    print(name, "=", round(coef, 5))