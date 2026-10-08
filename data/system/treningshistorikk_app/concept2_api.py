from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import requests


C2_ACCEPT_V1 = "application/vnd.c2logbook.v1+json"


@dataclass(frozen=True)
class Concept2ApiConfig:
    base_url: str
    access_token: str


class Concept2ApiError(RuntimeError):
    pass


def _auth_headers(access_token: str) -> dict[str, str]:
    return {
        "Accept": C2_ACCEPT_V1,
        "Authorization": f"Bearer {access_token}",
    }


def _get_json(url: str, headers: dict[str, str], params: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
    except requests.RequestException as exc:  # pragma: no cover
        raise Concept2ApiError(f"Nettverksfeil mot Concept2 API: {exc}") from exc

    if resp.status_code >= 400:
        try:
            payload: Any = resp.json()
        except ValueError:
            payload = {"message": resp.text}

        if isinstance(payload, dict):
            msg = payload.get("message") or payload.get("error_description") or str(payload)
        else:
            msg = str(payload)

        raise Concept2ApiError(f"API-feil {resp.status_code}: {msg}")

    try:
        data: Any = resp.json()
        if isinstance(data, dict):
            return data
        # Some endpoints/variants may return a bare list; wrap it for downstream.
        return {"data": data}
    except ValueError as exc:
        raise Concept2ApiError("Ugyldig JSON fra Concept2 API") from exc


def fetch_all_results(
    cfg: Concept2ApiConfig,
    *,
    from_date: date | None = None,
    to_date: date | None = None,
    type_filter: str | None = "skierg",
    per_page: int = 250,
) -> list[dict[str, Any]]:
    """Fetch all results for the authenticated user.

    Docs: GET /api/users/me/results
    - Supports query params: from, to, type, page, number
    - Pagination via meta.pagination.links.next
    """

    base = cfg.base_url.rstrip("/")
    url = f"{base}/api/users/me/results"

    params: dict[str, Any] = {"number": max(1, min(int(per_page), 250))}
    if from_date:
        params["from"] = from_date.isoformat()
    if to_date:
        params["to"] = to_date.isoformat()
    if type_filter:
        params["type"] = type_filter

    headers = _auth_headers(cfg.access_token)

    all_items: list[dict[str, Any]] = []
    next_url: str | None = url
    next_params: dict[str, Any] | None = params

    while next_url:
        payload = _get_json(next_url, headers=headers, params=next_params)

        data = payload.get("data")
        if not isinstance(data, list):
            raise Concept2ApiError("Uventet svar: 'data' er ikke en liste")
        all_items.extend(data)

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        pagination = meta.get("pagination")
        if not isinstance(pagination, dict):
            pagination = {}
        links = pagination.get("links")
        if not isinstance(links, dict):
            links = {}

        next_link = links.get("next")
        if next_link:
            # The API returns absolute next link; subsequent calls should not re-send params.
            next_url = str(next_link)
            next_params = None
        else:
            next_url = None

    return all_items


def fetch_result(
    cfg: Concept2ApiConfig,
    *,
    result_id: int,
    include: list[str] | None = None,
) -> dict[str, Any]:
    """Fetch a single result (more fields than the list endpoint).

    Docs: GET /api/users/me/results/{result_id}
    The API supports embedding via query param: ?include=strokes,metadata,user
    """

    base = cfg.base_url.rstrip("/")
    url = f"{base}/api/users/me/results/{int(result_id)}"

    params: dict[str, Any] | None = None
    if include:
        params = {"include": ",".join(include)}

    headers = _auth_headers(cfg.access_token)
    return _get_json(url, headers=headers, params=params)


def fetch_strokes(
    cfg: Concept2ApiConfig,
    *,
    result_id: int,
) -> list[dict[str, Any]]:
    """Fetch per-stroke data for a workout.

    Docs: GET /api/users/me/results/{result_id}/strokes
    Each stroke object contains:
      t   – time in tenths of a second (cumulative; resets to 0 for each interval
            in interval workouts)
      d   – distance in decimeters (cumulative; resets per interval)
      p   – pace in tenths of a second per 500 m
      spm – strokes per minute (current stroke rate)
      hr  – heart rate

    Raises Concept2ApiError on failure, including HTTP 404 when no stroke data
    exists for the workout.
    """
    base = cfg.base_url.rstrip("/")
    url = f"{base}/api/users/me/results/{int(result_id)}/strokes"
    headers = _auth_headers(cfg.access_token)
    payload = _get_json(url, headers=headers)
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    return data
