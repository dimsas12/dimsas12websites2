"""Approximate home-city coordinates of EuroLeague clubs (for travel distance)."""

from __future__ import annotations

import math

CITY = {
    "Madrid": (40.42, -3.70), "Barcelona": (41.39, 2.17), "Piraeus": (37.94, 23.65),
    "Athens": (37.98, 23.73), "Istanbul": (41.01, 28.98), "Vitoria": (42.85, -2.67),
    "Kaunas": (54.90, 23.90), "Tel Aviv": (32.08, 34.78), "Milan": (45.46, 9.19),
    "Belgrade": (44.79, 20.45), "Munich": (48.14, 11.58), "Moscow": (55.76, 37.62),
    "Lyon": (45.77, 4.88), "Valencia": (39.47, -0.38), "Berlin": (52.52, 13.40),
    "Monaco": (43.74, 7.42), "Bologna": (44.49, 11.34), "Malaga": (36.72, -4.42),
    "Bamberg": (49.89, 10.89), "Paris": (48.86, 2.35), "St Petersburg": (59.93, 30.34),
    "Dubai": (25.20, 55.27), "Kazan": (55.79, 49.12), "Krasnodar": (45.04, 38.98),
    "Zagreb": (45.81, 15.98), "Siena": (43.32, 11.33), "Podgorica": (42.44, 19.26),
    "Las Palmas": (28.12, -15.43), "Nizhny Novgorod": (56.33, 44.00),
    "Zielona Gora": (51.94, 15.51), "Strasbourg": (48.57, 7.75), "Sassari": (40.73, 8.56),
    "Limoges": (45.83, 1.26), "Vilnius": (54.69, 25.28), "Chalon": (46.78, 4.85),
    "Cantu": (45.74, 9.13), "Nanterre": (48.89, 2.21), "Izmir": (38.42, 27.14),
    "Ljubljana": (46.06, 14.51), "Klaipeda": (55.71, 21.14), "Kiev": (50.45, 30.52),
    "Gdynia": (54.52, 18.53), "Zgorzelec": (51.15, 15.01),
}

TEAM_CITY = {
    "MAD": "Madrid", "BAR": "Barcelona", "OLY": "Piraeus", "ULK": "Istanbul",
    "PAN": "Athens", "IST": "Istanbul", "BAS": "Vitoria", "ZAL": "Kaunas", "TEL": "Tel Aviv",
    "MIL": "Milan", "RED": "Belgrade", "MUN": "Munich", "CSK": "Moscow", "ASV": "Lyon",
    "PAM": "Valencia", "BER": "Berlin", "PAR": "Belgrade", "MCO": "Monaco", "VIR": "Bologna",
    "KHI": "Moscow", "MAL": "Malaga", "BAM": "Bamberg", "PRS": "Paris",
    "DYR": "St Petersburg", "DAR": "Istanbul", "GAL": "Istanbul", "HTA": "Tel Aviv",
    "DUB": "Dubai", "UNK": "Kazan", "BES": "Istanbul", "TIV": "Krasnodar", "CED": "Zagreb",
    "SIE": "Siena", "BUD": "Podgorica", "CAN": "Las Palmas", "NOV": "Nizhny Novgorod",
    "GSS": "Zielona Gora", "STR": "Strasbourg", "SAS": "Sassari", "LMG": "Limoges",
    "LIE": "Vilnius", "CHL": "Chalon", "CTU": "Cantu", "NTR": "Nanterre", "KSK": "Izmir",
    "LJU": "Ljubljana", "KLA": "Klaipeda", "NIK": "Kiev", "SOP": "Gdynia", "ZGO": "Zgorzelec",
}


def team_coords(team: str) -> tuple[float, float] | None:
    city = TEAM_CITY.get(team)
    return CITY.get(city) if city else None


def distance_km(a: str, b: str) -> float:
    """Great-circle distance between two clubs' home cities (0 if unknown)."""
    ca, cb = team_coords(a), team_coords(b)
    if not ca or not cb:
        return 0.0
    la1, lo1, la2, lo2 = map(math.radians, (*ca, *cb))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(h))
