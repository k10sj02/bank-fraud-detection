"""
watchman.py  —  Watchman OFAC / sanctions screening helper

Wraps the moov-io/watchman HTTP API for donor name screening.
Run Watchman locally:
    docker run -p 8084:8084 moov/watchman

Docs: https://moov-io.github.io/watchman/
"""

import os
from dataclasses import dataclass
from typing import Optional

import requests

# Default to local Docker instance; override via WATCHMAN_URL env var
WATCHMAN_URL = os.getenv("WATCHMAN_URL", "http://localhost:8084")

# Match score threshold — Watchman returns 0.0–1.0
# 0.75 is Watchman's own recommended minimum for compliance use
DEFAULT_MIN_MATCH = float(os.getenv("WATCHMAN_MIN_MATCH", "0.75"))

# Timeout for HTTP calls (seconds)
HTTP_TIMEOUT = int(os.getenv("WATCHMAN_TIMEOUT", "5"))


@dataclass
class WatchmanHit:
    """A single sanctions list match returned by Watchman."""
    entity_id:   str
    name:        str
    match_score: float
    list_name:   str          # e.g. "SDN", "EU-Consolidated", "UN-Consolidated"
    entity_type: str          # "individual" | "vessel" | "aircraft" | "organization"
    programs:    list[str]    # e.g. ["CUBA", "IRAN"]
    remarks:     str


@dataclass
class ScreeningResult:
    """Result of screening a single donor."""
    screened_name:   str
    screened_country: Optional[str]
    is_blocked:      bool          # True = hard block, do not process donation
    hits:            list[WatchmanHit]
    error:           Optional[str]  # Non-None if Watchman was unreachable
    watchman_live:   bool          # False = Watchman not running (graceful degradation)


def _is_alive() -> bool:
    """
    Check whether Watchman is reachable.
    v0.31.x uses /search (no version prefix) — /health returns 404.
    We probe /search directly since that's the only endpoint we need.
    """
    try:
        r = requests.get(f"{WATCHMAN_URL}/search",
                         params={"name": "ping", "limit": 1},
                         timeout=3)
        return r.status_code in (200, 400)
    except Exception:
        return False


def _api_path(path: str) -> str:
    """Return the correct search path for this Watchman version."""
    return "/search"  # v0.31.x — no version prefix


def screen_donor(
    name: str,
    country: Optional[str] = None,
    min_match: float = DEFAULT_MIN_MATCH,
    entity_type: str = "individual",
) -> ScreeningResult:
    """
    Screen a donor name against all loaded sanctions lists via Watchman.

    Parameters
    ----------
    name        : Donor full name
    country     : ISO-2 country code or plain name (optional — improves precision)
    min_match   : Minimum match score to treat as a hit (default 0.75)
    entity_type : "individual" | "organization" | "vessel" | "aircraft"

    Returns
    -------
    ScreeningResult with is_blocked=True if any hit exceeds min_match.
    If Watchman is unreachable, returns is_blocked=False with watchman_live=False
    so the ML pipeline can continue without hard-failing.
    """
    if not name or not name.strip():
        return ScreeningResult(
            screened_name="", screened_country=country,
            is_blocked=False, hits=[], error="Empty name", watchman_live=False,
        )

    if not _is_alive():
        return ScreeningResult(
            screened_name=name, screened_country=country,
            is_blocked=False, hits=[],
            error="Watchman not reachable — screening skipped. "
                  "Start with: docker run -p 8084:8084 moov/watchman",
            watchman_live=False,
        )

    params: dict = {
        "name":     name.strip(),
        "limit":    10,
        "minMatch": min_match,
        "type":     entity_type,
    }
    if country:
        params["country"] = country

    try:
        r = requests.get(
            f"{WATCHMAN_URL}/search",
            params=params,
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.Timeout:
        return ScreeningResult(
            screened_name=name, screened_country=country,
            is_blocked=False, hits=[],
            error=f"Watchman timed out after {HTTP_TIMEOUT}s",
            watchman_live=True,
        )
    except Exception as exc:
        return ScreeningResult(
            screened_name=name, screened_country=country,
            is_blocked=False, hits=[],
            error=f"Watchman error: {exc}",
            watchman_live=True,
        )

    hits: list[WatchmanHit] = []

    # Watchman v0.31.x returns a flat object with per-list keys:
    # { SDNs: [...], altNames: [...], euConsolidatedSanctionsList: [...], ... }
    LIST_MAP = {
        "SDNs":                      ("SDN",    "individual"),
        "altNames":                  ("SDN",    "individual"),
        "deniedPersons":             ("DPL",    "individual"),
        "bisEntities":               ("BIS",    "organization"),
        "euConsolidatedSanctionsList": ("EU-CSL", "individual"),
        "ukConsolidatedSanctionsList": ("UK-CSL", "individual"),
        "sectoralSanctions":         ("SSI",    "organization"),
    }

    for list_key, (list_name, default_type) in LIST_MAP.items():
        for entity in (data.get(list_key) or []):
            score = entity.get("match", 0.0)
            if score < min_match:
                continue
            # Each list has slightly different field names
            name = (
                entity.get("sdnName") or
                entity.get("alternateName") or
                entity.get("Name") or
                entity.get("name") or
                (entity.get("Names") or [""])[0]
            )
            programs = (
                entity.get("program") or
                entity.get("Programs") or
                []
            )
            remarks = (
                entity.get("remarks") or
                entity.get("Remarks") or
                (entity.get("OtherInfos") or [""])[0] or
                ""
            )
            entity_type = entity.get("sdnType") or entity.get("GroupType") or default_type
            hits.append(WatchmanHit(
                entity_id   = str(entity.get("entityID") or entity.get("EntityID") or entity.get("GroupID") or ""),
                name        = name,
                match_score = score,
                list_name   = list_name,
                entity_type = entity_type.lower() if entity_type else default_type,
                programs    = programs if isinstance(programs, list) else [programs],
                remarks     = str(remarks)[:300],
            ))

    # Sort highest match first
    hits.sort(key=lambda h: h.match_score, reverse=True)

    return ScreeningResult(
        screened_name    = name,
        screened_country = country,
        is_blocked       = len(hits) > 0,
        hits             = hits,
        error            = None,
        watchman_live    = True,
    )


def screen_bulk(
    donors: list[dict],
    min_match: float = DEFAULT_MIN_MATCH,
) -> dict[str, ScreeningResult]:
    """
    Screen a list of donors concurrently.

    Parameters
    ----------
    donors : list of dicts with keys: id, name, country (optional)

    Returns
    -------
    dict mapping donor id → ScreeningResult
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[str, ScreeningResult] = {}

    def _screen(donor: dict) -> tuple[str, ScreeningResult]:
        return donor["id"], screen_donor(
            name    = donor.get("name", ""),
            country = donor.get("country"),
            min_match = min_match,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_screen, d): d["id"] for d in donors}
        for future in as_completed(futures):
            donor_id, result = future.result()
            results[donor_id] = result

    return results


# ── Known OFAC names for demo / testing ──────────────────────────────────────
# These are real entries on the SDN list — useful for testing the integration
# when the dataset contains anonymised/fake donor names.
DEMO_SDN_NAMES = [
    "Nicolas Maduro",
    "Kim Jong Un",
    "Bashar al-Assad",
    "Viktor Bout",
    "Semion Mogilevich",
]


def inject_demo_hit(name: str) -> ScreeningResult:
    """
    Return a synthetic OFAC hit for demo purposes.
    Used when Watchman is live but the dataset has fake names.
    Call this in the scorer UI when the user types a known SDN name.
    """
    return ScreeningResult(
        screened_name    = name,
        screened_country = None,
        is_blocked       = True,
        hits             = [
            WatchmanHit(
                entity_id   = "DEMO-SDN-001",
                name        = name,
                match_score = 0.98,
                list_name   = "SDN (DEMO)",
                entity_type = "individual",
                programs    = ["SYRIA", "IRAN"],
                remarks     = "Demo hit — not a real Watchman result. "
                              "Replace with real donor data for production use.",
            )
        ],
        error          = None,
        watchman_live  = True,
    )
