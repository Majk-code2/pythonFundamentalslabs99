import pandas as pd

files = [
    "data/2022.csv",
    "data/2023.csv",
    "data/2024.csv",
    "data/2025.csv"
]

frames = []

for file in files:
    df = pd.read_csv(file)
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

data["tourney_date"] = pd.to_datetime(
    data["tourney_date"].astype(str),
    format="%Y%m%d"
)

data = data.sort_values("tourney_date")

print("===== TENNIS DATA =====")
print("Liczba meczow:", len(data))
print("Od:", data["tourney_date"].min())
print("Do:", data["tourney_date"].max())

print("\nMecze wedlug nawierzchni:")
print(data["surface"].value_counts())

print("\nBraki w rankingach:")
print("Winner rank:", data["winner_rank"].isna().sum())
print("Loser rank:", data["loser_rank"].isna().sum())