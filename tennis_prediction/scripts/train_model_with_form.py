import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

# Wczytanie danych
data = pd.read_csv("data/model_dataset_with_form.csv")

data["tourney_date"] = pd.to_datetime(data["tourney_date"])

# Trening: 2022-2024
train = data[data["tourney_date"].dt.year <= 2024].copy()

# Test: 2025
test = data[data["tourney_date"].dt.year == 2025].copy()

# Cechy modelu
features = [
    "rank_diff",
    "form5_diff"
]

X_train = train[features]
y_train = train["A_won"]

X_test = test[features]
y_test = test["A_won"]

# Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Wynik 0/1
predictions = model.predict(X_test)

# Prawdopodobienstwo zwyciestwa A
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
logloss = log_loss(y_test, probabilities)
brier = brier_score_loss(y_test, probabilities)

print("===== RANKING + FORMA =====")
print("Trening:", len(train))
print("Test:", len(test))
print()

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Log Loss:", round(logloss, 4))
print("Brier Score:", round(brier, 4))

print()
print("Porownanie:")
print("Ranking baseline: 64.40%")
print("Ranking ML:       64.04%")
print("Ranking + forma: ", round(accuracy * 100, 2), "%")

print()
print("Wagi modelu:")
for name, coef in zip(features, model.coef_[0]):
    print(name, "=", round(coef, 4))