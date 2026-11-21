import pandas as pd

# Wczytaj przygotowany dataset
data = pd.read_csv("data/model_dataset.csv")

# Zamiana daty
data["tourney_date"] = pd.to_datetime(data["tourney_date"])

# Interesuje nas rok 2025
test = data[data["tourney_date"].dt.year == 2025].copy()

# Niższy ranking ATP = lepszy zawodnik
test["predicted_A_won"] = (test["rank_A"] < test["rank_B"]).astype(int)

# Czy przewidywanie było poprawne?
test["correct"] = test["predicted_A_won"] == test["A_won"]

accuracy = test["correct"].mean()

print("===== RANKING BASELINE 2025 =====")
print("Liczba meczow:", len(test))
print("Poprawne:", test["correct"].sum())
print("Bledne:", (~test["correct"]).sum())
print("Accuracy:", round(accuracy * 100, 2), "%")