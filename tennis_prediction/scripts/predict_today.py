import os
import json
import requests

from dotenv import load_dotenv

# Plik .env znajduje się w katalogu tennis_prediction
load_dotenv()

api_key = os.getenv("CITO_API_KEY")

if not api_key:
    raise ValueError(
        "Brak CITO_API_KEY. Sprawdz plik .env."
    )

url = "https://api.citoapi.com/api/v1/tennis/odds/upcoming"

headers = {
    "x-api-key": api_key
}

try:
    response = requests.get(
        url,
        headers=headers,
        timeout=30
    )

    print("Status HTTP:", response.status_code)

    response.raise_for_status()

    data = response.json()

    print("\n===== ODPOWIEDZ CITO API =====")

    # Pokazujemy strukture odpowiedzi bez klucza API
    if isinstance(data, dict):
        print("Glowne pola:", list(data.keys()))

    elif isinstance(data, list):
        print("Liczba rekordow:", len(data))

    print("\nPierwszy fragment odpowiedzi:")

    print(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        )[:5000]
    )

except requests.exceptions.HTTPError as error:
    print("Blad HTTP:", error)
    print("Odpowiedz:", response.text[:1000])

except requests.exceptions.RequestException as error:
    print("Blad polaczenia:", error)

except ValueError as error:
    print("Blad danych JSON lub konfiguracji:", error)