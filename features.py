"""
features.py - Feature Engineering & Text Soup Construction

Key Engineering Decisions:
1. Entity Token Condensation:
   Names like "Christopher Nolan" or "Keanu Reeves" are stripped of whitespace and
   lowercased ("christophernolan", "keanureeves"). This prevents unigram tokenizers
   from treating "Christopher" and "Nolan" as independent tokens that inadvertently
   match other directors like "Christopher McQuarrie" or actors like "Nolan North".

2. Term Weighting via Frequency Multipliers:
   In TF-IDF, term frequency measures importance. By repeating key semantic signals
   in the composite "soup", we encode domain priorities:
   - Director (3x): Auteur style is one of the strongest predictors of cinematic affinity.
   - Top Cast (2x): Star appeal heavily influences audience preference.
   - Genres (2x): Ensures high baseline category alignment.
   - Keywords (1x): Provides topical context (e.g., "artificial intelligence", "dystopia").
   - Overview (1x): Provides narrative context and emotional vocabulary for matching.

3. Graceful Pandas Degradation:
   The functions return standard Python dictionaries and lists, and optionally create
   a pandas DataFrame when pandas is installed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore


def clean_token(text: str) -> str:
    """Lowercases, strips leading/trailing whitespace, and removes non-alphanumeric chars except space."""
    if not text:
        return ""
    text = text.lower().strip()
    return re.sub(r"[^\w\s-]", "", text)


def clean_entity_name(name: str) -> str:
    """
    Cleans an actor or director name and collapses whitespace.
    Example: 'Christopher Nolan' -> 'christophernolan'
    """
    if not name:
        return ""
    cleaned = re.sub(r"[^\w]", "", name.lower())
    return cleaned


def build_movie_soup(movie: Dict[str, Any]) -> str:
    """
    Constructs a weighted composite text 'soup' from movie metadata for TF-IDF vectorization.

    Weights:
      - Title: 2x (core title affinity)
      - Director: 3x (strong auteur signal)
      - Top Cast: 2x (lead performers)
      - Genres: 2x (categorical affinity)
      - Keywords: 1x (topical anchors)
      - Overview: 1x (natural language context)
    """
    soup_tokens: List[str] = []

    # 0. Title & Tagline (2x weight for title)
    title = movie.get("title") or ""
    if title and title != "Untitled":
        cleaned_title = clean_token(title)
        if cleaned_title:
            soup_tokens.extend([cleaned_title] * 2)

    tagline = movie.get("tagline") or ""
    if tagline:
        cleaned_tagline = clean_token(tagline)
        if cleaned_tagline:
            soup_tokens.append(cleaned_tagline)

    # 1. Director (3x weight)
    director = movie.get("director") or ""
    if director and director != "Unknown":
        d_token = clean_entity_name(director)
        if d_token:
            soup_tokens.extend([d_token] * 3)

    # Also support multiple directors if present
    for d in movie.get("directors", []):
        if d and d != director:
            dt = clean_entity_name(d)
            if dt:
                soup_tokens.extend([dt] * 2)

    # 2. Top Cast (2x weight for top 3, 1x for remaining)
    cast = movie.get("top_cast") or []
    for i, actor in enumerate(cast[:5]):
        a_token = clean_entity_name(actor)
        if a_token:
            weight = 2 if i < 3 else 1
            soup_tokens.extend([a_token] * weight)

    # 3. Genres (2x weight)
    genres = movie.get("genres") or []
    for g in genres:
        g_clean = clean_token(g).replace(" ", "")
        if g_clean:
            soup_tokens.extend([g_clean] * 2)

    # 4. Keywords (1x weight)
    keywords = movie.get("keywords") or []
    for kw in keywords:
        kw_clean = clean_token(kw).replace(" ", "")
        if kw_clean:
            soup_tokens.append(kw_clean)

    # 5. Overview (1x weight)
    overview = movie.get("overview") or ""
    if overview:
        cleaned_overview = clean_token(overview)
        soup_tokens.append(cleaned_overview)

    return " ".join(soup_tokens)


def enrich_movie_with_features(movie: Dict[str, Any]) -> Dict[str, Any]:
    """Adds engineered features including the text soup and normalized fields to a movie dict."""
    enriched = dict(movie)
    enriched["soup"] = build_movie_soup(enriched)

    # Decade feature (e.g., 1999 -> "1990s")
    year = enriched.get("year", 0)
    if year >= 1900:
        decade_num = (year // 10) * 10
        enriched["decade"] = f"{decade_num}s"
    else:
        enriched["decade"] = "Unknown"

    # Runtime bracket feature
    runtime = enriched.get("runtime", 0)
    if runtime <= 0:
        enriched["runtime_bracket"] = "Unknown"
    elif runtime < 90:
        enriched["runtime_bracket"] = "<90 min"
    elif runtime <= 120:
        enriched["runtime_bracket"] = "90-120 min"
    elif runtime <= 150:
        enriched["runtime_bracket"] = "120-150 min"
    else:
        enriched["runtime_bracket"] = "150+ min"

    return enriched


def prepare_features_dataframe(movies: List[Dict[str, Any]]) -> Any:
    """
    Enriches a list of movie dictionaries and returns a pandas DataFrame if pandas
    is available, otherwise returns the list of enriched dicts.
    """
    enriched = [enrich_movie_with_features(m) for m in movies]
    if pd is not None:
        return pd.DataFrame(enriched)
    return enriched
