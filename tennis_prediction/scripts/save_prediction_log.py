import pandas as pd

from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CURRENT_FILE = (
    ROOT
    / "data"
    / "current_predictions_elo.csv"
)

LOG_FILE = (
    ROOT
    / "data"
    / "prediction_log.csv"
)


if not CURRENT_FILE.exists():
    raise FileNotFoundError(
        "Brak data/current_predictions_elo.csv. "
        "Najpierw uruchom predict_current_matches_elo.py."
    )


if CURRENT_FILE.stat().st_size == 0:
    print("Raport predykcji jest pusty. Brak danych do zapisania.")
    raise SystemExit(0)

try:
    current = pd.read_csv(CURRENT_FILE)
except pd.errors.EmptyDataError:
    print("Raport predykcji nie ma kolumn. Brak danych do zapisania.")
    raise SystemExit(0)

if current.empty:
    print("Brak aktualnych predykcji do zapisania.")
    raise SystemExit(0)


required_columns = [
    "match_id",
    "start",
    "tournament",
    "surface",
    "competition_level",
    "player_A",
    "player_B",
    "predicted_winner",
    "model_probability",
    "fair_odds",
    "market_odds",
    "status",
]

missing = [
    column
    for column in required_columns
    if column not in current.columns
]

if missing:
    raise ValueError(
        f"Brak kolumn w raporcie: {missing}"
    )


now_utc = datetime.now(timezone.utc)

current["prediction_time_utc"] = (
    now_utc.isoformat()
)

current["model_version"] = (
    "logistic_surface_both_elo_v1"
)

current["start"] = pd.to_datetime(
    current["start"],
    utc=True,
    errors="coerce",
)


# Zapisujemy tylko prognozy wykonane przed startem meczu
current = current[
    current["start"] > now_utc
].copy()


if current.empty:
    print(
        "Brak przyszlych meczow. "
        "Nic nie zapisano."
    )
    raise SystemExit(0)


# Zachowujemy wszystkie obserwacje,
# także te poniżej progu 75%.
current["actual_winner"] = pd.NA
current["correct"] = pd.NA
current["closing_odds"] = pd.NA


if LOG_FILE.exists():

    log = pd.read_csv(LOG_FILE)

    # Nie zapisujemy ponownie tej samej predykcji
    existing_keys = set(
        zip(
            log["match_id"].astype(str),
            log["model_version"].astype(str),
        )
    )

    current = current[
        ~current.apply(
            lambda row: (
                str(row["match_id"]),
                str(row["model_version"]),
            )
            in existing_keys,
            axis=1,
        )
    ].copy()

else:

    log = pd.DataFrame()


if current.empty:
    print(
        "Te predykcje sa juz zapisane "
        "w prediction_log.csv."
    )
    raise SystemExit(0)


updated_log = pd.concat(
    [
        log,
        current,
    ],
    ignore_index=True,
)


updated_log.to_csv(
    LOG_FILE,
    index=False,
)


print("===== ZAPIS PREDYKCJI =====")
print("Nowe rekordy:", len(current))
print("Wszystkie rekordy w logu:", len(updated_log))
print("Zapisano:", LOG_FILE)

print(
    current[
        [
            "prediction_time_utc",
            "match_id",
            "start",
            "player_A",
            "player_B",
            "predicted_winner",
            "model_probability",
            "status",
        ]
    ].to_string(index=False)
)
