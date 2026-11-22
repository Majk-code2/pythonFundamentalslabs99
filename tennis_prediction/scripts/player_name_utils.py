import re
import unicodedata


def normalize_player_name(name):
    """
    Normalizuje zapis nazwiska zawodnika.

    Przyklady:
    S. D. Prajwal Dev -> s d prajwal dev
    S D Prajwal Dev   -> s d prajwal dev

    [merged into atp_208389] James Kent Trotter
    -> james kent trotter
    """

    if not isinstance(name, str):
        return ""

    # Usuwa informacje typu:
    # [merged into atp_208389]
    name = re.sub(
        r"\[merged into [^\]]+\]",
        "",
        name,
        flags=re.IGNORECASE
    )

    # Zamienia znaki akcentowane na podstawowe
    name = unicodedata.normalize("NFKD", name)

    name = "".join(
        character
        for character in name
        if not unicodedata.combining(character)
    )

    # Male litery
    name = name.lower()

    # Kropki zamieniamy na spacje
    name = name.replace(".", " ")

    # Myslniki i apostrofy jako spacje
    name = name.replace("-", " ")
    name = name.replace("'", " ")

    # Zostawia litery, cyfry i spacje
    name = re.sub(
        r"[^a-z0-9\s]",
        " ",
        name
    )

    # Usuwa wielokrotne spacje
    name = re.sub(
        r"\s+",
        " ",
        name
    ).strip()

    return name