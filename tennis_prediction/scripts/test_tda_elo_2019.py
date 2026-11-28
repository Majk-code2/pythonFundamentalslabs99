import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "tda_elo"
    / "elo_match_history_2019.csv"
)


def elo_probability(elo_a, elo_b):

    return 1.0 / (
        1.0 + 10 ** ((elo_b - elo_a) / 400.0)
    )


def evaluate(data, label):

    if len(data) < 10:
        print(label, "| za malo meczow")
        return

    y = data["A_won"].astype(int)

    models = {
        "OVERALL": data["p_overall"],
        "SURFACE": data["p_surface"],
        "COMBINED": data["p_combined"],
    }

    print("\n==============================")
    print(label)
    print("Mecze:", len(data))
    print("==============================")

    for name, probabilities in models.items():

        p = np.clip(
            probabilities.to_numpy(dtype=float),
            1e-6,
            1 - 1e-6,
        )

        predicted = (p >= 0.5).astype(int)

        accuracy = accuracy_score(
            y,
            predicted,
        )

        brier = brier_score_loss(
            y,
            p,
        )

        loss = log_loss(
            y,
            p,
            labels=[0, 1],
        )

        auc = roc_auc_score(
            y,
            p,
        ) if y.nunique() == 2 else np.nan

        print(f"\n{name}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Brier:    {brier:.4f}")
        print(f"Log Loss: {loss:.4f}")
        print(f"ROC AUC:  {auc:.4f}")


print("===== TEST TDA ELO 2019 =====")

data = pd.read_csv(INPUT)

data["start_time"] = pd.to_datetime(
    data["start_time"],
    errors="coerce",
    utc=True,
    format="mixed",
)

data = data[
    data["start_time"].dt.year == 2019
].copy()

print("Mecze 2019:", len(data))


# Neutralna orientacja zawodnikow.
# Player A jest wybierany wedlug player_id,
# a nie na podstawie wyniku meczu.

winner_is_a = (
    data["winner_id"].astype(str)
    < data["loser_id"].astype(str)
)

data["A_won"] = winner_is_a.astype(int)


def orient(winner_col, loser_col):

    return (
        np.where(
            winner_is_a,
            data[winner_col],
            data[loser_col],
        ),
        np.where(
            winner_is_a,
            data[loser_col],
            data[winner_col],
        ),
    )


overall_a, overall_b = orient(
    "winner_overall_elo",
    "loser_overall_elo",
)

surface_a, surface_b = orient(
    "winner_surface_elo",
    "loser_surface_elo",
)


data["p_overall"] = elo_probability(
    overall_a,
    overall_b,
)

data["p_surface"] = elo_probability(
    surface_a,
    surface_b,
)


# Pierwsza kombinacja: 50% Overall + 50% Surface.
# Nie jest to jeszcze wytrenowana kalibracja.

data["p_combined"] = (
    0.5 * data["p_overall"]
    + 0.5 * data["p_surface"]
)


# Podzial na poziomy rozgrywek.

data["segment"] = data["tier"].replace({
    "tour": "ATP",
    "slam": "ATP",
    "challenger": "CHALLENGER",
    "itf": "ITF",
})


# Wymagamy poprawnych danych Elo.

required = [
    "p_overall",
    "p_surface",
    "p_combined",
]

data = data.dropna(
    subset=required
).copy()


print("Mecze do testu:", len(data))


evaluate(
    data,
    "WSZYSTKIE MECZE",
)

for segment in [
    "ATP",
    "CHALLENGER",
    "ITF",
]:

    subset = data[
        data["segment"] == segment
    ]

    evaluate(
        subset,
        segment,
    )


# Dodatkowo sprawdzamy mecze,
# w ktorych obaj zawodnicy mieli historie.

experienced = data[
    (data["winner_matches_before"] >= 10)
    &
    (data["loser_matches_before"] >= 10)
]

evaluate(
    experienced,
    "OBAJ ZAWODNICY MINIMUM 10 MECZOW",
)


print("\n===== KONIEC TESTU =====")