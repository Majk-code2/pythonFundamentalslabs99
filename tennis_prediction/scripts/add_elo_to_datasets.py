import pandas as pd
import numpy as np

from pathlib import Path

from player_name_utils import normalize_player_name


ROOT = Path(__file__).resolve().parents[1]

SURFACE_DATA = (
    ROOT
    / "data"
    / "model_dataset_surface.csv"
)

ELO_DATA = (
    ROOT
    / "data"
    / "elo_surface_elo_history.csv"
)


print("===== WCZYTYWANIE DANYCH =====")

surface = pd.read_csv(SURFACE_DATA)
elo = pd.read_csv(ELO_DATA)

surface["tourney_date"] = pd.to_datetime(
    surface["tourney_date"]
)

elo["tourney_date"] = pd.to_datetime(
    elo["tourney_date"]
)


# Normalizacja nazw Player A i Player B
surface["player_A_key"] = (
    surface["player_A"]
    .apply(normalize_player_name)
)

surface["player_B_key"] = (
    surface["player_B"]
    .apply(normalize_player_name)
)


# Tworzymy niezalezny od kolejnosci klucz pary zawodnikow
surface["pair_key"] = surface.apply(
    lambda row: "|||".join(
        sorted(
            [
                row["player_A_key"],
                row["player_B_key"]
            ]
        )
    ),
    axis=1
)

elo["pair_key"] = elo.apply(
    lambda row: "|||".join(
        sorted(
            [
                row["winner_key"],
                row["loser_key"]
            ]
        )
    ),
    axis=1
)


# Klucze laczenia
merge_columns = [
    "tourney_date",
    "tourney_id",
    "round",
    "surface",
    "competition_level",
    "pair_key"
]


elo_columns = merge_columns + [
    "winner_key",
    "loser_key",
    "winner_elo_before",
    "loser_elo_before",
    "winner_surface_elo_before",
    "loser_surface_elo_before"
]


# Usuwamy ewentualne powtorzenia tego samego klucza
elo_lookup = (
    elo[elo_columns]
    .drop_duplicates(
        subset=merge_columns,
        keep="last"
    )
)


print("Dataset modelu:", len(surface))
print("Historia Elo:", len(elo))
print("Lookup Elo:", len(elo_lookup))


# Laczenie
merged = surface.merge(
    elo_lookup,
    on=merge_columns,
    how="left",
    validate="many_to_one"
)


# Czy Player A byl historycznym zwyciezca?
a_is_winner = (
    merged["player_A_key"]
    == merged["winner_key"]
)


# Elo Player A i Player B przed meczem
merged["elo_A"] = np.where(
    a_is_winner,
    merged["winner_elo_before"],
    merged["loser_elo_before"]
)

merged["elo_B"] = np.where(
    a_is_winner,
    merged["loser_elo_before"],
    merged["winner_elo_before"]
)


# Surface Elo Player A i Player B
merged["surface_elo_A"] = np.where(
    a_is_winner,
    merged["winner_surface_elo_before"],
    merged["loser_surface_elo_before"]
)

merged["surface_elo_B"] = np.where(
    a_is_winner,
    merged["loser_surface_elo_before"],
    merged["winner_surface_elo_before"]
)


# Cechy dla modelu
merged["elo_diff"] = (
    merged["elo_A"]
    - merged["elo_B"]
)

merged["surface_elo_diff"] = (
    merged["surface_elo_A"]
    - merged["surface_elo_B"]
)


# Sprawdzenie dopasowania
matched = merged["elo_diff"].notna().sum()
missing = merged["elo_diff"].isna().sum()

print("\n===== DOPASOWANIE =====")
print("Dopasowane mecze:", matched)
print("Brak Elo:", missing)
print(
    "Pokrycie:",
    round(
        matched / len(merged) * 100,
        2
    ),
    "%"
)


# Mecze bez dopasowania usuwamy z eksperymentu Elo
model_data = merged.dropna(
    subset=[
        "elo_diff",
        "surface_elo_diff"
    ]
).copy()


# Usuwamy pomocnicze kolumny
columns_to_remove = [
    "pair_key",
    "winner_key",
    "loser_key",
    "winner_elo_before",
    "loser_elo_before",
    "winner_surface_elo_before",
    "loser_surface_elo_before"
]

model_data = model_data.drop(
    columns=columns_to_remove,
    errors="ignore"
)


# Podzial ATP / Challenger
atp = model_data[
    model_data["competition_level"] == "ATP"
].copy()

challenger = model_data[
    model_data["competition_level"]
    == "CHALLENGER"
].copy()


atp_output = (
    ROOT
    / "data"
    / "model_dataset_surface_elo_atp.csv"
)

challenger_output = (
    ROOT
    / "data"
    / "model_dataset_surface_elo_challenger.csv"
)


atp.to_csv(
    atp_output,
    index=False
)

challenger.to_csv(
    challenger_output,
    index=False
)


print("\n===== WYNIK =====")
print("ATP:", len(atp))
print("Challenger:", len(challenger))

print("\nNowe cechy:")
print(
    model_data[
        [
            "player_A",
            "player_B",
            "elo_A",
            "elo_B",
            "elo_diff",
            "surface_elo_A",
            "surface_elo_B",
            "surface_elo_diff",
            "A_won"
        ]
    ]
    .tail(10)
    .to_string(index=False)
)

print("\nZapisano:")
print(atp_output)
print(challenger_output)