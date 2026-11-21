import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Wczytanie danych
data = pd.read_csv("data/model_dataset.csv")

data["tourney_date"] = pd.to_datetime(data["tourney_date"])

# Cecha: roznica rankingu
data["rank_diff"] = data["rank_B"] - data["rank_A"]

# Przeszlosc = trening
train = data[data["tourney_date"].dt.year <= 2024].copy()

# Przyszlosc = test
test = data[data["tourney_date"].dt.year == 2025].copy()

X_train = train[["rank_diff"]]
y_train = train["A_won"]

X_test = test[["rank_diff"]]
y_test = test["A_won"]

# Model
model = LogisticRegression()

model.fit(X_train, y_train)

# Predykcje
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("===== LOGISTIC REGRESSION =====")
print("Trening:", len(train), "meczow")
print("Test:", len(test), "meczow")
print("Accuracy:", round(accuracy * 100, 2), "%")
print()
print("Baseline rankingu: 64.4%")