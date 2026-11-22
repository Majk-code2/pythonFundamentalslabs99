from player_name_utils import normalize_player_name


test_names = [
    "S. D. Prajwal Dev",
    "S D Prajwal Dev",
    "[merged into atp_208389] James Kent Trotter",
    "James Kent Trotter",
    "Matheus Pucinelli de Almeida",
    "Matheus Almeida",
    "Thiago Seyboth Wild",
    "Thiago Seyboth-Wild",
    "Borna Ćorić",
    "Borna Coric"
]


print("===== TEST NORMALIZACJI =====")

for name in test_names:

    normalized = normalize_player_name(name)

    print(
        f"{name:55} -> {normalized}"
    )