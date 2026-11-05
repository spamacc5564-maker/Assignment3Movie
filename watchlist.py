"""
watchlist.py - Session-Based Watchlist, Taste Profiling & CSV Persistence

Key Design Decisions:
1. Session-State Centric:
   Stores the user's watchlist in Streamlit's `st.session_state["watchlist"]` as a dictionary
   keyed by movie ID for O(1) membership lookups and fast toggle updates.

2. User Taste Profiling (Centroid Aggregation):
   Aggregates the user's saved movies into a composite 'taste vector':
   - Combines the text soups of all watchlisted movies into a collective user preference document.
   - Computes empirical genre proportions, favorite eras/decades, and top directors.
   - Matches candidate discovery movies against this taste centroid using TF-IDF cosine similarity.

3. Cross-Session Persistence via CSV Export & Import:
   Allows users to download their curated library to a CSV file and restore it in future
   sessions using `st.download_button` and `st.file_uploader`.
"""

from __future__ import annotations

import io
import csv
import logging
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

from features import build_movie_soup
from nlp import compute_tfidf_matrix, compute_vector_similarity

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore

logger = logging.getLogger(__name__)

# Fallback in-memory storage for test runners or non-Streamlit environments
_IN_MEMORY_WATCHLIST: Dict[int, Dict[str, Any]] = {}


def _get_storage() -> Dict[int, Dict[str, Any]]:
    """Returns the active storage dict (st.session_state when in Streamlit, otherwise in-memory)."""
    global _IN_MEMORY_WATCHLIST
    if st is not None:
        try:
            if "watchlist" not in st.session_state:
                st.session_state["watchlist"] = {}
            return st.session_state["watchlist"]
        except Exception:
            pass
    return _IN_MEMORY_WATCHLIST


def init_watchlist() -> None:
    """Ensures watchlist data structure is initialized."""
    _get_storage()


def get_watchlist() -> List[Dict[str, Any]]:
    """Returns the current list of watchlisted movie dictionaries."""
    storage = _get_storage()
    return list(storage.values())


def is_in_watchlist(movie_id: int) -> bool:
    """Checks whether a movie is currently saved in the user's watchlist."""
    storage = _get_storage()
    return movie_id in storage


def add_to_watchlist(movie: Dict[str, Any]) -> bool:
    """Adds a movie to the session watchlist. Returns True if added."""
    storage = _get_storage()
    mid = movie.get("id")
    if mid:
        storage[mid] = movie
        return True
    return False


def remove_from_watchlist(movie_id: int) -> bool:
    """Removes a movie from the session watchlist. Returns True if removed."""
    storage = _get_storage()
    if movie_id in storage:
        del storage[movie_id]
        return True
    return False


def toggle_watchlist(movie: Dict[str, Any]) -> bool:
    """Toggles a movie's inclusion in the watchlist. Returns new state (True=saved)."""
    mid = movie.get("id", 0)
    if is_in_watchlist(mid):
        remove_from_watchlist(mid)
        return False
    else:
        add_to_watchlist(movie)
        return True


def clear_watchlist() -> None:
    """Empties the current session watchlist."""
    storage = _get_storage()
    storage.clear()


def compute_taste_profile(watchlist: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyzes the user's saved movies to build a comprehensive taste profile.
    Returns:
      {
        "total_movies": int,
        "favorite_genres": List[Tuple[str, int, float]],
        "favorite_decades": List[Tuple[str, int]],
        "top_directors": List[Tuple[str, int]],
        "avg_rating": float,
        "avg_runtime": int,
        "composite_soup": str,
      }
    """
    if not watchlist:
        return {
            "total_movies": 0,
            "favorite_genres": [],
            "favorite_decades": [],
            "top_directors": [],
            "avg_rating": 0.0,
            "avg_runtime": 0,
            "composite_soup": "",
        }

    total = len(watchlist)
    genre_counter = Counter()
    decade_counter = Counter()
    director_counter = Counter()
    total_rating = 0.0
    total_runtime = 0

    soups: List[str] = []

    for m in watchlist:
        for g in m.get("genres", []):
            genre_counter[g] += 1

        year = m.get("year", 0)
        if year >= 1900:
            dec = f"{(year // 10) * 10}s"
            decade_counter[dec] += 1

        d = m.get("director")
        if d and d != "Unknown":
            director_counter[d] += 1

        total_rating += float(m.get("vote_average", 0.0))
        total_runtime += int(m.get("runtime", 0))

        soups.append(build_movie_soup(m))

    favorite_genres = [
        (g, count, round((count / total) * 100, 1))
        for g, count in genre_counter.most_common(5)
    ]
    favorite_decades = decade_counter.most_common(4)
    top_directors = director_counter.most_common(4)
    avg_rating = round(total_rating / total, 1)
    avg_runtime = int(total_runtime / total) if total > 0 else 0
    composite_soup = " ".join(soups)

    return {
        "total_movies": total,
        "favorite_genres": favorite_genres,
        "favorite_decades": favorite_decades,
        "top_directors": top_directors,
        "avg_rating": avg_rating,
        "avg_runtime": avg_runtime,
        "composite_soup": composite_soup,
    }


def recommend_from_profile(
    watchlist: List[Dict[str, Any]],
    candidate_pool: List[Dict[str, Any]],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Generates personalized recommendations by matching candidates against the
    user's aggregate taste profile centroid.
    """
    if not watchlist or not candidate_pool:
        return []

    saved_ids = set(m.get("id") for m in watchlist)
    pool = [m for m in candidate_pool if m.get("id") not in saved_ids]
    if not pool:
        return []

    profile = compute_taste_profile(watchlist)
    user_soup = profile["composite_soup"]
    if not user_soup.strip():
        return []

    pool_soups = [build_movie_soup(m) for m in pool]
    all_docs = [user_soup] + pool_soups

    vectorizer, matrix = compute_tfidf_matrix(all_docs)

    # User profile is doc 0
    from nlp import compute_similarity
    scores = compute_similarity(vectorizer, matrix, target_idx=0)[1:]

    results: List[Dict[str, Any]] = []
    for i, m in enumerate(pool):
        sim = scores[i]
        rec = dict(m)
        rec["profile_match_score"] = round(sim, 3)
        rec["match_score"] = round(sim * 100)
        rec["match_pct"] = f"{round(sim * 100)}%"
        results.append(rec)

    results.sort(key=lambda x: x["profile_match_score"], reverse=True)
    return results[:top_n]


def export_watchlist_csv(watchlist: List[Dict[str, Any]]) -> str:
    """Exports the watchlist to a CSV formatted string for easy downloading."""
    output = io.StringIO()
    fieldnames = [
        "id", "title", "year", "runtime", "certification",
        "genres", "director", "vote_average", "vote_count",
        "poster", "overview"
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for m in watchlist:
        row = dict(m)
        row["genres"] = ", ".join(m.get("genres", []))
        writer.writerow(row)

    return output.getvalue()


def import_watchlist_csv(csv_content: str) -> Tuple[int, List[str]]:
    """
    Parses a CSV string and merges valid movie records into the current session watchlist.
    Returns (num_imported: int, error_messages: List[str]).
    """
    init_watchlist()
    errors: List[str] = []
    imported_count = 0

    try:
        reader = csv.DictReader(io.StringIO(csv_content))
        for row in reader:
            mid_raw = row.get("id")
            if not mid_raw or not mid_raw.strip().isdigit():
                continue

            mid = int(mid_raw.strip())
            title = row.get("title") or "Untitled"
            year = int(row.get("year", 0)) if row.get("year", "0").isdigit() else 0
            runtime = int(row.get("runtime", 0)) if row.get("runtime", "0").isdigit() else 0

            genres_raw = row.get("genres", "")
            genres = [g.strip() for g in genres_raw.split(",") if g.strip()]

            movie_obj = {
                "id": mid,
                "title": title,
                "year": year,
                "runtime": runtime,
                "certification": row.get("certification", "NR"),
                "genres": genres,
                "director": row.get("director", "Unknown"),
                "vote_average": float(row.get("vote_average", 0.0)),
                "vote_count": int(row.get("vote_count", 0)) if row.get("vote_count", "0").isdigit() else 0,
                "poster": row.get("poster") or "",
                "overview": row.get("overview") or "",
            }

            storage = _get_storage()
            storage[mid] = movie_obj
            imported_count += 1

    except Exception as e:
        errors.append(f"Failed to parse CSV file: {str(e)}")

    return imported_count, errors
