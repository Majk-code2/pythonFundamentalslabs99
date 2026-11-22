import pandas as pd
from pathlib import Path


FILES = [
    Path("data/2022.csv"),
    Path("data/2022_challenger.csv"),
    Path("data/2023.csv"),
    Path("data/2023_challenger.csv"),
    Path("data/2024.csv"),
    Path("data/2024_challenger.csv"),
    Path("data/2025.csv"),
    Path("data/2025_challenger.csv"),
    Path("data/2026.csv"),
    Path("data/2026_challenger.csv"),
]


REQUIRED = [
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_bpSaved",
    "w_bpFaced",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_bpSaved",
    "l_bpFaced",
]


rows = []


for file_path in FILES:

    data = pd.read_csv(file_path)

    missing_columns = [
        column
        for column in REQUIRED
        if column not in data.columns
    ]

    if missing_columns:

        rows.append(
            {
                "file": file_path.name,
                "matches": len(data),
                "complete_return_data": 0,
                "coverage_pct": 0.0,
                "missing_columns": ", ".join(
                    missing_columns
                ),
            }
        )

        continue

    complete = data.dropna(
        subset=REQUIRED
    )

    rows.append(
        {
            "file": file_path.name,
            "matches": len(data),
            "complete_return_data": len(complete),
            "coverage_pct": (
                len(complete)
                / len(data)
                * 100
            ),
            "missing_columns": "",
        }
    )


result = pd.DataFrame(rows)


print("===== POKRYCIE DANYCH RETURNOWYCH =====")

print(
    result.to_string(
        index=False,
        formatters={
            "coverage_pct": (
                lambda value: f"{value:.2f}%"
            )
        },
    )
)


total_matches = result["matches"].sum()

total_complete = (
    result["complete_return_data"].sum()
)


print("\n===== RAZEM =====")

print("Mecze:", total_matches)

print(
    "Kompletne dane:",
    total_complete,
)

print(
    "Pokrycie:",
    round(
        total_complete
        / total_matches
        * 100,
        2,
    ),
    "%",
)