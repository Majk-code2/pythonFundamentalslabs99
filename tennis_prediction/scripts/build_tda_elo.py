import json
import glob
import argparse

from pathlib import Path
from collections import defaultdict

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "tda_archive"
OUTPUT_DIR = ROOT / "data" / "tda_elo"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


START_ELO = 1500.0
K_FACTOR = 24.0

ALLOWED_TIERS = {
    "tour",
    "slam",
    "challenger",
    "itf",
}

SURFACE_MAP = {
    "hard": "hard",
    "hard_indoor": "hard",
    "clay": "clay",
    "clay_indoor": "clay",
    "grass": "grass",
    "grass_indoor": "grass",
}

# Pierwszy wariant: jednakowa waga.
# Rozne wagi poziomow przetestujemy pozniej.
TIER_WEIGHTS = {
    "tour": 1.0,
    "slam": 1.0,
    "challenger": 1.0,
    "itf": 1.0,
}


def expected_score(elo_a, elo_b):

    return 1.0 / (
        1.0 + 10 ** ((elo_b - elo_a) / 400.0)
    )


def load_matches(end_year):

    files = []

    for year in range(2018, end_year + 1):

        pattern = str(
            DATA_DIR / f"tda_{year}-*.json"
        )

        files.extend(glob.glob(pattern))

    files = sorted(files)

    print("Pliki:", len(files))

    matches = []
    seen_ids = set()

    skipped = defaultdict(int)

    for file_path in files:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            daily_matches = json.load(file)

        for match in daily_matches:

            match_id = match.get("id")

            if not match_id:
                skipped["missing_match_id"] += 1
                continue

            if match_id in seen_ids:
                skipped["duplicate_match_id"] += 1
                continue

            seen_ids.add(match_id)

            if match.get("event_type") != "singles":
                skipped["not_singles"] += 1
                continue

            if match.get("status") != "completed":
                skipped["not_completed"] += 1
                continue

            tournament = (
                match.get("tournament") or {}
            )

            tier = tournament.get("tier")

            if tier not in ALLOWED_TIERS:
                skipped["unsupported_tier"] += 1
                continue

            raw_surface = tournament.get("surface")

            surface = SURFACE_MAP.get(raw_surface)

            if not surface:
                skipped["unknown_surface"] += 1
                continue

            winner = match.get("winner") or {}
            loser = match.get("loser") or {}

            winner_id = winner.get("id")
            loser_id = loser.get("id")

            if not winner_id or not loser_id:
                skipped["missing_player_id"] += 1
                continue

            if winner_id == loser_id:
                skipped["same_player"] += 1
                continue

            start_time = (
                match.get("start_time")
                or match.get("date")
            )

            if not start_time:
                skipped["missing_date"] += 1
                continue

            matches.append({
                "match_id": match_id,
                "start_time": start_time,
                "tournament_id": tournament.get("id"),
                "tier": tier,
                "surface": surface,
                "raw_surface": raw_surface,
                "winner_id": winner_id,
                "loser_id": loser_id,
                "winner_name": winner.get("name"),
                "loser_name": loser.get("name"),
            })

    print("Mecze zakwalifikowane:", len(matches))
    print("Pominiete:", dict(skipped))

    return matches


def build_elo(matches, end_year):

    data = pd.DataFrame(matches)

    if data.empty:
        raise ValueError("Brak meczow do obliczenia Elo")

    data["start_time"] = pd.to_datetime(
        data["start_time"],
        utc=True,
        errors="coerce",
        format="mixed",
    )

    data = data.dropna(
        subset=["start_time"]
    ).copy()

        # Wszystkie mecze jednego dnia traktujemy
    # jako jedna grupe chronologiczna.
    data["match_date"] = data["start_time"].dt.date

    data = data.sort_values(
        ["match_date", "match_id"]
    ).reset_index(drop=True)

    ratings = defaultdict(
        lambda: {
            "overall": START_ELO,
            "hard": START_ELO,
            "clay": START_ELO,
            "grass": START_ELO,
            "matches": 0,
        }
    )

    player_names = {}

    history_rows = []

    # Wszystkie mecze jednego dnia korzystaja
    # z Elo sprzed tego dnia.
    for match_date, group in data.groupby(
        "match_date",
        sort=True
    ):

        updates = []

        for match in group.itertuples(index=False):

            winner_id = match.winner_id
            loser_id = match.loser_id

            surface = match.surface
            tier = match.tier

            winner_rating = ratings[winner_id]
            loser_rating = ratings[loser_id]

            player_names[winner_id] = match.winner_name
            player_names[loser_id] = match.loser_name

            winner_overall = winner_rating["overall"]
            loser_overall = loser_rating["overall"]

            winner_surface = winner_rating[surface]
            loser_surface = loser_rating[surface]

            expected_overall = expected_score(
                winner_overall,
                loser_overall
            )

            expected_surface = expected_score(
                winner_surface,
                loser_surface
            )

            k = K_FACTOR * TIER_WEIGHTS[tier]

            delta_overall = k * (
                1.0 - expected_overall
            )

            delta_surface = k * (
                1.0 - expected_surface
            )

            history_rows.append({
                "match_id": match.match_id,
                "start_time": match_date,
                "tier": tier,
                "surface": surface,
                "winner_id": winner_id,
                "loser_id": loser_id,
                "winner_overall_elo": winner_overall,
                "loser_overall_elo": loser_overall,
                "winner_surface_elo": winner_surface,
                "loser_surface_elo": loser_surface,
                "winner_matches_before": winner_rating["matches"],
                "loser_matches_before": loser_rating["matches"],
                "expected_winner_overall": expected_overall,
                "expected_winner_surface": expected_surface,
            })

            updates.append((
                winner_id,
                loser_id,
                surface,
                delta_overall,
                delta_surface,
            ))

        # Aktualizacja dopiero po zapisaniu
        # cech przedmeczowych calej grupy.
        for (
            winner_id,
            loser_id,
            surface,
            delta_overall,
            delta_surface,
        ) in updates:

            ratings[winner_id]["overall"] += delta_overall
            ratings[loser_id]["overall"] -= delta_overall

            ratings[winner_id][surface] += delta_surface
            ratings[loser_id][surface] -= delta_surface

            ratings[winner_id]["matches"] += 1
            ratings[loser_id]["matches"] += 1

    snapshot_rows = []

    for player_id, rating in ratings.items():

        snapshot_rows.append({
            "player_id": player_id,
            "player_name": player_names.get(player_id),
            "overall_elo": rating["overall"],
            "hard_elo": rating["hard"],
            "clay_elo": rating["clay"],
            "grass_elo": rating["grass"],
            "matches": rating["matches"],
        })

    snapshot = pd.DataFrame(snapshot_rows)

    snapshot = snapshot.sort_values(
        "overall_elo",
        ascending=False
    )

    history = pd.DataFrame(history_rows)

    snapshot_path = (
        OUTPUT_DIR / f"elo_snapshot_{end_year}.csv"
    )

    history_path = (
        OUTPUT_DIR / f"elo_match_history_{end_year}.csv"
    )

    snapshot.to_csv(
        snapshot_path,
        index=False
    )

    history.to_csv(
        history_path,
        index=False
    )

    print("\n===== TDA ELO =====")
    print("Rok koncowy:", end_year)
    print("Mecze:", len(history))
    print("Zawodnicy:", len(snapshot))

    print("\nTOP 20 OVERALL ELO")

    print(
        snapshot[
            [
                "player_name",
                "overall_elo",
                "hard_elo",
                "clay_elo",
                "matches",
            ]
        ].head(20).round(2).to_string(index=False)
    )

    print("\nZapisano:")
    print(snapshot_path)
    print(history_path)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--end-year",
        type=int,
        default=2018
    )

    args = parser.parse_args()

    if args.end_year < 2018:
        raise ValueError("Rok musi byc >= 2018")

    matches = load_matches(args.end_year)

    build_elo(matches, args.end_year)


if __name__ == "__main__":
    main()