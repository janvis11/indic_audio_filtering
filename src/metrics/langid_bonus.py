from __future__ import annotations

from typing import Optional

# Map manifest language names to expected 2-letter / common codes
LANG_NAME_TO_CODES = {
    "assamese": {"as"},
    "bengali": {"bn"},
    "bodo": {"brx", "bd"},
    "dogri": {"doi", "dgo"},
    "gujarati": {"gu"},
    "hindi": {"hi"},
    "kannada": {"kn"},
    "kashmiri": {"ks"},
    "konkani": {"gom"},
    "maithili": {"mai"},
    "malayalam": {"ml"},
    "manipuri": {"mni"},
    "marathi": {"mr"},
    "nepali": {"ne"},
    "odia": {"or", "od"},
    "punjabi": {"pa", "pan"},
    "sanskrit": {"sa"},
    "santali": {"sat"},
    "sindhi": {"sd"},
    "tamil": {"ta"},
    "telugu": {"te"},
    "urdu": {"ur"},
}

def _normalize_expected_language(expected_language: Optional[str]) -> set[str]:
    if not expected_language:
        return set()

    lang = expected_language.strip().lower()

    if lang in LANG_NAME_TO_CODES:
        return LANG_NAME_TO_CODES[lang]

    # If the manifest already contains a short code, keep it
    if len(lang) <= 3:
        return {lang}

    return {lang[:2]}

def _normalize_predicted_language(predicted_language: Optional[str]) -> Optional[str]:
    if not predicted_language:
        return None
    return predicted_language.strip().lower()

def compute_language_match(expected_language: Optional[str], predicted_language: Optional[str]) -> dict:
    """
    Soft language consistency check.

    Returns:
        language_match: bool | None
        language_score: float | None

    Notes:
    - If expected language is unavailable, do not penalize.
    - If predicted language is unavailable, do not penalize.
    - This is meant to be a weak auxiliary signal, not a hard gate.
    """
    expected_codes = _normalize_expected_language(expected_language)
    predicted = _normalize_predicted_language(predicted_language)

    if not expected_codes or not predicted:
        return {
            "language_match": None,
            "language_score": None,
        }

    match = predicted in expected_codes
    return {
        "language_match": bool(match),
        "language_score": 1.0 if match else 0.0,
    }