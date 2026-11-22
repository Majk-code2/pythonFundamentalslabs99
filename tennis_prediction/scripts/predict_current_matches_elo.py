import pandas as pd
import numpy as np
from player_name_utils import normalize_player_name

from pathlib import Path
from collections import defaultdict, deque
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]

HISTORY_FILES = HISTORY_FILES = [
    ROOT / "data" / "2022.csv",
    ROOT / "data" / "2022_challenger.csv",
    ROOT / "data" / "2023.csv",
    ROOT / "data" / "2023_challenger.csv",
    ROOT / "data" / "2024.csv",
    ROOT / "data" / "2024_challenger.csv",
    ROOT / "data" / "2025.csv",
    ROOT / "data" / "2025_challenger.csv",
    ROOT / "data" / "2026.csv",
    ROOT / "data" / "2026_challenger.csv"
]

UPCOMING_FILE = (
    ROOT
    / "data"
    / "upcoming_atp_with_surface.csv"
)

MODEL_DATA_FILES = {
    "ATP": (
        ROOT
        / "data"
        / "model_dataset_surface_elo_atp.csv"
    ),

    "CHALLENGER": (
        ROOT
        / "data"
        / "model_dataset_surface_elo_challenger.csv"
    )
}


FEATURES = [
    "rank_diff",
    "surface_win_diff",
    "surface_ace_diff",
    "surface_df_diff",
    "surface_first_won_diff",
    "surface_second_won_diff",
    "surface_bp_saved_diff",
    "elo_diff",
    "surface_elo_diff"
]


def safe_divide(a, b):

    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan

    return a / b


def create_match_stats(row, prefix, won):

    service_points = row.get(f"{prefix}_svpt")
    first_in = row.get(f"{prefix}_1stIn")

    if pd.notna(service_points) and pd.notna(first_in):
        second_serve_points = service_points - first_in
    else:
        second_serve_points = np.nan

    return {
        "won": won,

        "ace_rate": safe_divide(
            row.get(f"{prefix}_ace"),
            service_points
        ),

        "df_rate": safe_divide(
            row.get(f"{prefix}_df"),
            service_points
        ),

        "first_won_rate": safe_divide(
            row.get(f"{prefix}_1stWon"),
            first_in
        ),

        "second_won_rate": safe_divide(
            row.get(f"{prefix}_2ndWon"),
            second_serve_points
        ),

        "bp_saved_rate": safe_divide(
            row.get(f"{prefix}_bpSaved"),
            row.get(f"{prefix}_bpFaced")
        )
    }


def summarize_history(matches):

    if not matches:

        return {
            "matches": 0,
            "win_rate": 0.5,
            "ace_rate": 0.0,
            "df_rate": 0.0,
            "first_won_rate": 0.0,
            "second_won_rate": 0.0,
            "bp_saved_rate": 0.0
        }

    frame = pd.DataFrame(list(matches))

    return {
        "matches": len(frame),

        "win_rate": frame["won"].mean(),

        "ace_rate": frame["ace_rate"].mean(),

        "df_rate": frame["df_rate"].mean(),

        "first_won_rate": (
            frame["first_won_rate"].mean()
        ),

        "second_won_rate": (
            frame["second_won_rate"].mean()
        ),

        "bp_saved_rate": (
            frame["bp_saved_rate"].mean()
        )
    }


print("===== BUDOWANIE AKTUALNEJ HISTORII =====")


# Historia: gracz -> nawierzchnia -> ostatnie 10 meczow
history = defaultdict(
    lambda: defaultdict(
        lambda: deque(maxlen=10)
    )
)


historical_data = pd.concat(
    [
        pd.read_csv(file)
        for file in HISTORY_FILES
    ],
    ignore_index=True
)


historical_data["tourney_date"] = pd.to_datetime(
    historical_data["tourney_date"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)


historical_data = historical_data.dropna(
    subset=[
        "tourney_date",
        "surface",
        "winner_name",
        "loser_name"
    ]
).copy()


historical_data = historical_data[
    historical_data["surface"].isin(
        ["Hard", "Clay", "Grass"]
    )
].copy()


historical_data = historical_data.sort_values(
    [
        "tourney_date",
        "tourney_id",
        "match_num"
    ]
)
# Budujemy aktualny stan historii zawodnikow
for _, match in historical_data.iterrows():
    surface = match["surface"]
    winner = normalize_player_name(match["winner_name"])
    loser = normalize_player_name(match["loser_name"])

    winner_stats = create_match_stats(
        match,
        "w",
        1
    )

    loser_stats = create_match_stats(
        match,
        "l",
        0
    )

    history[winner][surface].append(
        winner_stats
    )

    history[loser][surface].append(
        loser_stats
    )


print(
"Historyczni zawodnicy:",
len(history)
)

print(
    "Historyczni zawodnicy:",
    len(history)
)

print("\n===== TRENOWANIE MODELI =====")

models = {}

for competition_level, model_file in MODEL_DATA_FILES.items():

    model_data = pd.read_csv(model_file)

    model_data[FEATURES] = (
        model_data[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    X_train = model_data[FEATURES]
    y_train = model_data["A_won"]

    model = LogisticRegression(
        max_iter=2000
    )

    model.fit(
        X_train,
        y_train
    )

    models[competition_level] = model

    print(
        competition_level,
        "- mecze treningowe:",
        len(model_data)
    )


print("\n===== WCZYTYWANIE AKTUALNEGO ELO =====")

elo_snapshot = pd.read_csv(
    ROOT / "data" / "current_elo_snapshot.csv"
)

elo_lookup = elo_snapshot.set_index(
    "player_key"
)

print(
    "Zawodnicy w snapshot Elo:",
    len(elo_snapshot)
)

print("\n===== AKTUALNE MECZE =====")


print("\n===== AKTUALNE MECZE =====")


upcoming = pd.read_csv(UPCOMING_FILE)

from datetime import datetime, timezone

upcoming["start"] = pd.to_datetime(
    upcoming["start"],
    utc=True,
    errors="coerce"
)

now_utc = datetime.now(timezone.utc)

upcoming = upcoming[
    upcoming["start"] > now_utc
].copy()

upcoming = upcoming[
    upcoming["surface"].isin(
        ["Hard", "Clay", "Grass"]
    )
].copy()


predictions = []


for _, match in upcoming.iterrows():

    player_a = match["player_A"]
    player_b = match["player_B"]

    player_a_key = normalize_player_name(player_a)
    player_b_key = normalize_player_name(player_b)


    surface = match["surface"]

    competition_level = str(
        match["competition_level"]
    ).strip().upper()

    model = models.get(competition_level)

    if model is None:
        print(
            "Pominieto mecz bez odpowiedniego modelu:",
            player_a,
            "vs",
            player_b,
            "| poziom:",
            competition_level
        )
        continue

    stats_a = summarize_history(
        history[player_a_key][surface]
    )

    stats_b = summarize_history(
        history[player_b_key][surface]
    )

    rank_a = match["rank_A"]
    rank_b = match["rank_B"]

    if (
        player_a_key not in elo_lookup.index
        or player_b_key not in elo_lookup.index
    ):
        print(
            "Pominieto mecz bez Elo:",
            player_a,
            "vs",
            player_b
        )
        continue

    elo_a = float(
        elo_lookup.loc[
            player_a_key,
            "overall_elo"
        ]
    )

    elo_b = float(
        elo_lookup.loc[
            player_b_key,
            "overall_elo"
        ]
    )

    surface_column = (
        f"{str(surface).strip().lower()}_elo"
    )

    if surface_column not in elo_lookup.columns:
        print(
            "Pominieto mecz z nieznana nawierzchnia:",
            surface
        )
        continue

    surface_elo_a = float(
        elo_lookup.loc[
            player_a_key,
            surface_column
        ]
    )

    surface_elo_b = float(
        elo_lookup.loc[
            player_b_key,
            surface_column
        ]
    )

    feature_row = pd.DataFrame(
        [
            {
                "rank_diff": rank_b - rank_a,

                "surface_win_diff": (
                    stats_a["win_rate"]
                    - stats_b["win_rate"]
                ),

                "surface_ace_diff": (
                    stats_a["ace_rate"]
                    - stats_b["ace_rate"]
                ),

                "surface_df_diff": (
                    stats_a["df_rate"]
                    - stats_b["df_rate"]
                ),

                "surface_first_won_diff": (
                    stats_a["first_won_rate"]
                    - stats_b["first_won_rate"]
                ),

                "surface_second_won_diff": (
                    stats_a["second_won_rate"]
                    - stats_b["second_won_rate"]
                ),

                "surface_bp_saved_diff": (
                    stats_a["bp_saved_rate"]
                    - stats_b["bp_saved_rate"]
                ),

                "elo_diff": (
                    elo_a - elo_b
                ),

                "surface_elo_diff": (
                    surface_elo_a
                    - surface_elo_b
                )
            }
        ]
    )


    feature_row = (
        feature_row[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


    probability_a = (
        model.predict_proba(feature_row)[0, 1]
    )

    probability_b = 1.0 - probability_a


    if probability_a >= probability_b:

        predicted_winner = player_a
        model_probability = probability_a
        winner_odds = match.get("odds_A")

    else:

        predicted_winner = player_b
        model_probability = probability_b
        winner_odds = match.get("odds_B")


    # Minimalna ilosc historii dla obu zawodnikow
    enough_history = (
        stats_a["matches"] >= 5
        and stats_b["matches"] >= 5
    )


    # Przygotowanie kursu i porownania z rynkiem
    market_probability = None
    probability_edge = None
    fair_odds = None

    # Fair odds wynikajace z prawdopodobienstwa modelu
    if model_probability > 0:
        fair_odds = 1.0 / model_probability

    # Akceptujemy tylko wiarygodny zakres kursow
    valid_market_odds = (
        pd.notna(winner_odds)
        and 1.01 <= winner_odds <= 20.00
    )

    if valid_market_odds:

        market_probability = (
            1.0 / winner_odds
        )

        probability_edge = (
            model_probability
            - market_probability
        )

    else:

        # Podejrzany lub brakujacy kurs,
        # np. 100.00, nie jest uzywany
        winner_odds = None
        market_probability = None
        probability_edge = None

    # Klasyfikacja predykcji
    if not enough_history:
        status = "POMIN: za malo historii"

    elif model_probability >= 0.85:
        status = "MOCNY KANDYDAT"

    elif model_probability >= 0.75:
        status = "KANDYDAT"

    elif model_probability >= 0.65:
        status = "OBSERWACJA"

    else:
        status = "POMIN: niska pewnosc"


    predictions.append(
        {
            "match_id": match.get("match_id"),
            "start": match.get("start"),
            "tournament": match.get("tournament"),
            "surface": surface,
            "competition_level": competition_level,

            "elo_A": elo_a,
            "elo_B": elo_b,
            "elo_diff": elo_a - elo_b,

            "surface_elo_A": surface_elo_a,
            "surface_elo_B": surface_elo_b,
            "surface_elo_diff": (
                surface_elo_a - surface_elo_b
            ),

            "player_A": player_a,
            "rank_A": rank_a,
            "history_A": stats_a["matches"],

            "player_B": player_b,
            "rank_B": rank_b,
            "history_B": stats_b["matches"],

            "predicted_winner": predicted_winner,
            "model_probability": model_probability,

            "fair_odds": fair_odds,
            "market_odds": winner_odds,
            "market_probability_raw": market_probability,
            "probability_edge_raw": probability_edge,

            "status": status
        }
    )

results = pd.DataFrame(predictions)

if results.empty:
    print("\n===== PODSUMOWANIE =====")
    print("Brak meczow zakwalifikowanych do analizy.")
    print(
        "Mozliwe powody: mecz juz rozpoczęty, "
        "brak Elo, brak nawierzchni albo brak poziomu modelu."
    )

    output_path = (
        ROOT
        / "data"
        / "current_predictions_elo.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("Zapisano pusty raport:", output_path)
    raise SystemExit(0)


results = results.sort_values(
    "model_probability",
    ascending=False
)


for _, row in results.iterrows():

    print("\n--------------------------------------")
    print("Start:", row["start"])
    print("Turniej:", row["tournament"])
    print("Nawierzchnia:", row["surface"])
    print("Poziom modelu:", row["competition_level"])

    print(
        f"{row['player_A']} "
        f"(#{int(row['rank_A'])}, "
        f"historia {int(row['history_A'])})"
    )

    print(
        f"vs {row['player_B']} "
        f"(#{int(row['rank_B'])}, "
        f"historia {int(row['history_B'])})"
    )

    print(
        "Overall Elo:",
        f"{row['elo_A']:.1f}",
        "-",
        f"{row['elo_B']:.1f}"
    )

    print(
        "Surface Elo:",
        f"{row['surface_elo_A']:.1f}",
        "-",
        f"{row['surface_elo_B']:.1f}"
    )

    print(
        "Model wskazuje:",
        row["predicted_winner"]
    )

    print(
        "Prawdopodobienstwo modelu:",
        f"{row['model_probability'] * 100:.2f}%"
    )

    print(
        "Fair odds modelu:",
        (
            f"{row['fair_odds']:.2f}"
            if pd.notna(row["fair_odds"])
            else "brak"
        )
    )

    print(
        "Kurs rynkowy:",
        (
            f"{row['market_odds']:.2f}"
            if pd.notna(row["market_odds"])
            else "brak"
        )
    )

    print(
        "Status:",
        row["status"]
    )


output_path = (
    ROOT
    / "data"
    / "current_predictions_elo.csv"
)

results.to_csv(
    output_path,
    index=False
)


print("\n===== PODSUMOWANIE =====")

print(
    "Wszystkie analizowane mecze:",
    len(results)
)

print(
    "Minimum 75% i wystarczajaca historia:",
    len(
        results[
            results["status"].isin(
                [
                    "KANDYDAT",
                    "MOCNY KANDYDAT"
                ]
            )
        ]
    )
)

print(
    "Zapisano:",
    output_path
)