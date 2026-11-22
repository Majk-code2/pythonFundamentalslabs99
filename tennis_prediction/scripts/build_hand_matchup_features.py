import pandas as pd

from collections import defaultdict, deque
from pathlib import Path

from player_name_utils import normalize_player_name


ROOT = Path(__file__).resolve().parents[1]


FILES = [
    (ROOT / "data" / "2022.csv", "ATP"),
    (ROOT / "data" / "2022_challenger.csv", "CHALLENGER"),
    (ROOT / "data" / "2023.csv", "ATP"),
    (ROOT / "data" / "2023_challenger.csv", "CHALLENGER"),
    (ROOT / "data" / "2024.csv", "ATP"),
    (ROOT / "data" / "2024_challenger.csv", "CHALLENGER"),
    (ROOT / "data" / "2025.csv", "ATP"),
    (ROOT / "data" / "2025_challenger.csv", "CHALLENGER"),
    (ROOT / "data" / "2026.csv", "ATP"),
    (ROOT / "data" / "2026_challenger.csv", "CHALLENGER"),
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


def normalize_hand(value):

    if not isinstance(value, str):
        return None

    value = value.strip().upper()

    if value in ["L", "R"]:
        return value

    return None


def summarize(records, opponent_hand):

    relevant = [
        record["won"]
        for record in records
        if record["opponent_hand"] == opponent_hand
    ]

    if not relevant:
        return {
            "matches": 0,
            "win_rate": 0.5,
        }

    return {
        "matches": len(relevant),
        "win_rate": sum(relevant) / len(relevant),
    }


frames = []

for file_path, competition_level in FILES:

    frame = pd.read_csv(file_path)

    frame["competition_level"] = competition_level

    frames.append(frame)


data = pd.concat(
    frames,
    ignore_index=True,
)


data["tourney_date"] = pd.to_datetime(
    data["tourney_date"].astype(str),
    format="%Y%m%d",
    errors="coerce",
)


data["winner_hand_clean"] = (
    data["winner_hand"].apply(normalize_hand)
)

data["loser_hand_clean"] = (
    data["loser_hand"].apply(normalize_hand)
)


data = data.dropna(
    subset=[
        "tourney_date",
        "tourney_id",
        "round",
        "surface",
        "winner_name",
        "loser_name",
        "winner_hand_clean",
        "loser_hand_clean",
    ]
).copy()


data = data[
    data["surface"].isin(
        ["Hard", "Clay", "Grass"]
    )
].copy()


data["round_order"] = (
    data["round"].map(ROUND_ORDER)
)


data = data.dropna(
    subset=["round_order"]
).copy()


data["round_order"] = (
    data["round_order"].astype(int)
)


data["winner_key"] = (
    data["winner_name"]
    .apply(normalize_player_name)
)

data["loser_key"] = (
    data["loser_name"]
    .apply(normalize_player_name)
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
    lambda: deque(maxlen=30)
)


rows = []


for (
    tourney_date,
    tourney_id,
), tournament in data.groupby(
    [
        "tourney_date",
        "tourney_id",
    ],
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

            winner_hand = match[
                "winner_hand_clean"
            ]

            loser_hand = match[
                "loser_hand_clean"
            ]


            winner_before = summarize(
                history[winner_key],
                loser_hand,
            )

            loser_before = summarize(
                history[loser_key],
                winner_hand,
            )


            pair_key = "|||".join(
                sorted(
                    [
                        winner_key,
                        loser_key,
                    ]
                )
            )


            rows.append(
                {
                    "tourney_date": tourney_date,
                    "tourney_id": tourney_id,
                    "round": match["round"],
                    "surface": match["surface"],
                    "competition_level": (
                        match["competition_level"]
                    ),
                    "pair_key": pair_key,

                    "winner_key": winner_key,
                    "loser_key": loser_key,

                    "winner_hand": winner_hand,
                    "loser_hand": loser_hand,

                    "winner_matches_vs_hand": (
                        winner_before["matches"]
                    ),

                    "loser_matches_vs_hand": (
                        loser_before["matches"]
                    ),

                    "winner_win_rate_vs_hand": (
                        winner_before["win_rate"]
                    ),

                    "loser_win_rate_vs_hand": (
                        loser_before["win_rate"]
                    ),
                }
            )


            updates.append(
                (
                    winner_key,
                    {
                        "opponent_hand": loser_hand,
                        "won": 1,
                    },
                )
            )

            updates.append(
                (
                    loser_key,
                    {
                        "opponent_hand": winner_hand,
                        "won": 0,
                    },
                )
            )


        for player_key, record in updates:

            history[player_key].append(record)


hand_features = pd.DataFrame(rows)


output_path = (
    ROOT
    / "data"
    / "hand_matchup_features.csv"
)


hand_features.to_csv(
    output_path,
    index=False,
)


print("===== HAND MATCHUP FEATURES =====")
print("Mecze:", len(hand_features))

print(
    "Mecze L vs R:",
    (
        hand_features["winner_hand"]
        != hand_features["loser_hand"]
    ).sum(),
)

print(
    "Mecze ta sama reka:",
    (
        hand_features["winner_hand"]
        == hand_features["loser_hand"]
    ).sum(),
)

print(
    "Srednia historia winner vs hand:",
    round(
        hand_features[
            "winner_matches_vs_hand"
        ].mean(),
        2,
    ),
)

print(
    "Srednia historia loser vs hand:",
    round(
        hand_features[
            "loser_matches_vs_hand"
        ].mean(),
        2,
    ),
)

print("\nZapisano:")
print(output_path)