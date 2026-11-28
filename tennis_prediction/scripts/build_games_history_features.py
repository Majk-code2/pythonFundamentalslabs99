import re
import numpy as np
import pandas as pd

from pathlib import Path
from collections import defaultdict, deque

from player_name_utils import normalize_player_name


ROOT = Path(__file__).resolve().parents[1]

FILES = [
    (ROOT / "data" / f"{year}.csv", "ATP")
    for year in range(2022, 2027)
] + [
    (
        ROOT / "data" / f"{year}_challenger.csv",
        "CHALLENGER",
    )
    for year in range(2022, 2027)
]


SET_RE = re.compile(
    r"^(\d{1,2})-(\d{1,2})(?:\(\d+\))?$"
)

INVALID = [
    "RET",
    "W/O",
    "DEF",
    "ABD",
    "ABN",
]


ROUND_ORDER = {
    "Q1": 1,
    "Q2": 2,
    "Q3": 3,
    "R128": 4,
    "R64": 5,
    "R32": 6,
    "RR": 7,
    "R16": 8,
    "QF": 9,
    "SF": 10,
    "F": 11,
    "BR": 12,
}


def parse_score(score):

    if not isinstance(score, str):
        return None

    score = score.strip().upper()

    if any(x in score for x in INVALID):
        return None

    tokens = score.split()

    if len(tokens) not in (2, 3):
        return None

    total_games = 0
    tiebreaks = 0
    winner_sets = 0
    loser_sets = 0

    for token in tokens:

        match = SET_RE.fullmatch(token)

        if match is None:
            return None

        a = int(match.group(1))
        b = int(match.group(2))

        total_games += a + b

        winner_sets += int(a > b)
        loser_sets += int(b > a)

        if max(a, b) == 7 and min(a, b) == 6:
            tiebreaks += 1

    if winner_sets != 2:
        return None

    if loser_sets not in [0, 1]:
        return None

    return {
        "total_games": total_games,
        "sets_played": len(tokens),
        "tiebreaks": tiebreaks,
    }


def summarize(records):

    if not records:
        return {
            "games_history_matches": 0,
            "avg_total_games": np.nan,
            "avg_games_per_set": np.nan,
            "tiebreak_rate": np.nan,
            "three_set_rate": np.nan,
        }

    frame = pd.DataFrame(list(records))

    return {
        "games_history_matches": len(frame),
        "avg_total_games": frame["total_games"].mean(),
        "avg_games_per_set": frame["games_per_set"].mean(),
        "tiebreak_rate": frame["has_tiebreak"].mean(),
        "three_set_rate": frame["three_sets"].mean(),
    }


frames = []

for path, level in FILES:

    frame = pd.read_csv(path)
    frame["competition_level"] = level

    frames.append(frame)


data = pd.concat(
    frames,
    ignore_index=True,
)

data = data[data["best_of"].eq(3)].copy()

data["parsed"] = data["score"].apply(parse_score)

data = data.dropna(subset=["parsed"]).copy()

data["total_games"] = data["parsed"].apply(
    lambda x: x["total_games"]
)

data["sets_played"] = data["parsed"].apply(
    lambda x: x["sets_played"]
)

data["tiebreaks"] = data["parsed"].apply(
    lambda x: x["tiebreaks"]
)

data["games_per_set"] = (
    data["total_games"] / data["sets_played"]
)

data["has_tiebreak"] = (
    data["tiebreaks"] > 0
).astype(int)

data["three_sets"] = (
    data["sets_played"] == 3
).astype(int)


data["tourney_date"] = pd.to_datetime(
    data["tourney_date"].astype(str),
    format="%Y%m%d",
    errors="coerce",
)

data["round_order"] = data["round"].map(
    ROUND_ORDER
)

data = data.dropna(
    subset=[
        "tourney_date",
        "round_order",
        "surface",
        "winner_name",
        "loser_name",
        "match_num",
    ]
).copy()


data["winner_key"] = data["winner_name"].apply(
    normalize_player_name
)

data["loser_key"] = data["loser_name"].apply(
    normalize_player_name
)


data = data.sort_values(
    [
        "tourney_date",
        "tourney_id",
        "round_order",
        "match_num",
    ]
).reset_index(drop=True)


history = defaultdict(
    lambda: deque(maxlen=10)
)

rows = []


for (
    tourney_date,
    tourney_id,
), tournament in data.groupby(
    ["tourney_date", "tourney_id"],
    sort=True,
):

    for _, round_matches in tournament.groupby(
        "round_order",
        sort=True,
    ):

        updates = []

        for _, match in round_matches.iterrows():

            winner_key = match["winner_key"]
            loser_key = match["loser_key"]

            winner_before = summarize(
                history[winner_key]
            )

            loser_before = summarize(
                history[loser_key]
            )

            pair_key = "|||".join(
                sorted([winner_key, loser_key])
            )

            row = {
                "tourney_date": tourney_date,
                "tourney_id": match["tourney_id"],
                "match_num": match["match_num"],
                "round": match["round"],
                "surface": match["surface"],
                "competition_level": match["competition_level"],
                "pair_key": pair_key,
            }

            for prefix, summary in [
                ("winner", winner_before),
                ("loser", loser_before),
            ]:

                for key, value in summary.items():
                    row[f"{prefix}_{key}"] = value

            rows.append(row)

            record = {
                "total_games": match["total_games"],
                "games_per_set": match["games_per_set"],
                "has_tiebreak": match["has_tiebreak"],
                "three_sets": match["three_sets"],
            }

            updates.append((winner_key, record))
            updates.append((loser_key, record))

        for player_key, record in updates:
            history[player_key].append(record)


result = pd.DataFrame(rows)


MERGE_KEYS = [
    "tourney_date",
    "tourney_id",
    "match_num",
    "round",
    "surface",
    "competition_level",
    "pair_key",
]


duplicates = result.duplicated(
    MERGE_KEYS,
    keep=False,
)

print("===== GAMES HISTORY =====")
print("Mecze:", len(result))
print("Duplikaty klucza:", duplicates.sum())


if duplicates.any():

    print(
        result.loc[
            duplicates,
            MERGE_KEYS
        ].to_string(index=False)
    )

    raise ValueError(
        "Duplikaty pozostaja mimo match_num"
    )


output = (
    ROOT
    / "data"
    / "games_history_features.csv"
)

result.to_csv(
    output,
    index=False
)

print("Zapisano:", output)

print("\nOstatnie rekordy:")
print(result.tail(5).to_string(index=False))