from __future__ import annotations

import json
from pathlib import Path

PERSON_IDS = [
    "M003", "M005", "M007", "M009", "M011", "M012", "M013", "M019",
    "M022", "M023", "M024", "M025", "M026", "M027", "M028", "M029",
    "M030", "M031", "M032", "M033", "M034", "M035", "M037", "M039",
    "M040", "M041", "M042", "W009", "W011", "W014", "W015", "W016",
    "W018", "W019", "W023", "W024", "W025", "W026", "W028", "W029",
    "W033", "W035", "W036", "W037", "W038", "W040",
]


def ensure_identity_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(PERSON_IDS, indent=2), encoding="utf-8")


def one_hot(person_id: str) -> list[float]:
    if person_id not in PERSON_IDS:
        raise ValueError(f"Unknown MEAD identity: {person_id}")
    vector = [0.0] * len(PERSON_IDS)
    vector[PERSON_IDS.index(person_id)] = 1.0
    return vector

