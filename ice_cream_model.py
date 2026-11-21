import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Wczytanie danych
data = pd.read_csv("ice-cream.csv")

# 2. Zamiana tekstu DayOfWeek i Month na dane liczbowe
data = pd.get_dummies(data, columns=["DayOfWeek", "Month"])

# 3. X = dane, na podstawie których model przewiduje
X = data.drop(columns=["Date", "IceCreamsSold"])

# y = wartość, którą model ma przewidzieć
y = data["IceCreamsSold"]

# 4. Podział danych: 70% nauka, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# 5. Tworzymy model
model = LinearRegression()

# 6. Model się uczy
model.fit(X_train, y_train)

# 7. Model przewiduje wyniki dla danych testowych
predictions = model.predict(X_test)

# 8. Sprawdzamy jakość modelu
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("===== WYNIKI MODELU =====")
print("MAE:", mae)
print("R2:", r2)

# 9. Znajdź najzimniejszy dzień
original_data = pd.read_csv("ice-cream.csv")

coldest_day = original_data.loc[original_data["Temperature"].idxmin()]

print("\n===== NAJZIMNIEJSZY DZIEN =====")
print("Data:", coldest_day["Date"])
print("Dzień tygodnia:", coldest_day["DayOfWeek"])
print("Miesiąc:", coldest_day["Month"])
print("Temperatura:", coldest_day["Temperature"])
print("Opady:", coldest_day["Rainfall"])
print("Rzeczywiście sprzedano:", coldest_day["IceCreamsSold"])

# Przygotuj najzimniejszy dzień w takim samym formacie jak dane treningowe
coldest_input = data.loc[[original_data["Temperature"].idxmin()]].drop(
    columns=["Date", "IceCreamsSold"]
)

# Predykcja modelu
predicted_sales = model.predict(coldest_input)[0]

print("\n===== PREDYKCJA MODELU =====")
print("Rzeczywista sprzedaż:", coldest_day["IceCreamsSold"])
print("Model przewiduje:", round(predicted_sales, 1))
print("Błąd:", round(abs(coldest_day["IceCreamsSold"] - predicted_sales), 1))

# 10. Znajdź największy błąd modelu na danych testowych

results = original_data.loc[X_test.index].copy()

results["Predicted"] = predictions
results["Error"] = abs(results["IceCreamsSold"] - results["Predicted"])

worst = results.loc[results["Error"].idxmax()]

print("\n===== NAJWIĘKSZY BŁĄD MODELU =====")
print("Data:", worst["Date"])
print("Dzień tygodnia:", worst["DayOfWeek"])
print("Miesiąc:", worst["Month"])
print("Temperatura:", worst["Temperature"])
print("Opady:", worst["Rainfall"])
print("Rzeczywiście sprzedano:", worst["IceCreamsSold"])
print("Model przewidział:", round(worst["Predicted"], 1))
print("Błąd:", round(worst["Error"], 1))
print("\n===== TOP 5 NAJWIĘKSZYCH BŁĘDÓW =====")

top5 = results.sort_values("Error", ascending=False).head(5)

print(
    top5[
        [
            "Date",
            "DayOfWeek",
            "Month",
            "Temperature",
            "Rainfall",
            "IceCreamsSold",
            "Predicted",
            "Error"
        ]
    ]
)