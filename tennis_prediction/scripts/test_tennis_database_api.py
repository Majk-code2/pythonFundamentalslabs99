import os
import requests

from pathlib import Path
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")

API_KEY = os.getenv(
    "TENNIS_DATABASE_API_KEY"
)

if not API_KEY:
    raise ValueError(
        "Brak TENNIS_DATABASE_API_KEY w .env"
    )


BASE_URL = (
    "https://api.tennisdatabaseapi.com/v1"
)


headers = {
    "Authorization": f"Bearer {API_KEY}"
}


url = (
    BASE_URL
    + "/results/yesterday"
)


params = {
    "tour": "men",
}


response = requests.get(
    url,
    headers=headers,
    params=params,
    timeout=30,
)


print("HTTP:", response.status_code)


if response.ok:

    payload = response.json()

    data = payload.get(
        "data",
        []
    )

    meta = payload.get(
        "meta",
        {}
    )

    print("Mecze:", len(data))
    print("Meta:", meta)

    print("\nPierwszy rekord:")

    if data:
        print(data[0])

else:

    print(
        response.text[:1000]
    )