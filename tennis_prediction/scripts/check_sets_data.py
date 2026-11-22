import re
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


INVALID_MARKERS = [
    "RET",
    "W/O",
    "WO",
    "DEF",
    "ABD",
    "ABN",
    "CANCELLED",
]


SET_PATTERN = re.compile(
    r"^\d{1,2}-\d{1,2}(?:\(\d+\))?$"
)


def classify_score(score):

    if not isinstance(score, str):
        return None

    score = score.strip().upper()

    if not score:
        return None

    if any(
        marker in score
        for marker in INVALID_MARKERS
    ):
        return None

    tokens = score.split()

    valid_sets = [
        token
        for token in tokens
        if SET_PATTERN.match(token)
    ]

    if len(valid_sets) == 2:
        return 0

    if len(valid_sets) == 3:
        return 1

    return None


rows = []


for file_path in FILES:

    data = pd.read_csv(file_path)

    best_of_3 = data[
        data["best_of"] == 3
    ].copy()

    best_of_3["over_2_5_sets"] = (
        best_of_3["score"]
        .apply(classify_score)
    )

    valid = best_of_3.dropna(
        subset=["over_2_5_sets"]
    ).copy()

    valid["over_2_5_sets"] = (
        valid["over_2_5_sets"]
        .astype(int)
    )

    under_count = (
        valid["over_2_5_sets"]
        .eq(0)
        .sum()
    )

    over_count = (
        valid["over_2_5_sets"]
        .eq(1)
        .sum()
    )

    rows.append(
        {
            "file": file_path.name,
            "all_matches": len(data),
            "best_of_3": len(best_of_3),
            "valid_scores": len(valid),
            "under_2_5": under_count,
            "over_2_5": over_count,
            "over_pct": (
                over_count / len(valid) * 100
                if len(valid) > 0
                else 0
            ),
        }
    )


result = pd.DataFrame(rows)


print("===== DANE OVER/UNDER 2,5 SETA =====")

print(
    result.to_string(
        index=False,
        formatters={
            "over_pct": (
                lambda value: f"{value:.2f}%"
            )
        },
    )
)


total_matches = result["all_matches"].sum()
total_bo3 = result["best_of_3"].sum()
total_valid = result["valid_scores"].sum()
total_under = result["under_2_5"].sum()
total_over = result["over_2_5"].sum()


print("\n===== RAZEM =====")
print("Wszystkie mecze:", total_matches)
print("Best of 3:", total_bo3)
print("Poprawne wyniki setowe:", total_valid)
print("Under 2,5:", total_under)
print("Over 2,5:", total_over)

print(
    "Procent Over 2,5:",
    round(
        total_over / total_valid * 100,
        2,
    )
    if total_valid > 0
    else 0,
    "%",
)