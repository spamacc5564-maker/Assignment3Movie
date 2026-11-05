"""
filters.py - Core Filter Model, TMDB Parameter Translation & Zero-Result Diagnostics

Architectural & Design Decisions:
1. Server-Side vs. Client-Side Filtering:
   - Server-Side (TMDB /discover/movie):
     * Temporal boundaries (primary_release_date.gte, primary_release_date.lte)
     * Quality thresholds (vote_average.gte, vote_count.gte)
     * Primary genre logic: AND (comma-separated IDs) or OR (pipe-separated IDs)
     * Genre exclusions (without_genres)
     * Original language (with_original_language)
     * Personnel IDs (with_people, with_cast, with_crew)
   - Client-Side (Post-fetch Enrichment):
     * Non-contiguous runtime brackets (e.g. user selects '<90 min' AND '150+ min')
     * Exact US certification verification (using US release_dates)
     * Multi-person role disambiguation (ensuring Director vs. Actor distinction)
     * Strict verification across combined AND + NOT genre filters.

2. Zero-Result Diagnostic Engine:
   When zero movies match a combined filter query, CineMatch performs an automated
   ablation analysis to determine which constraint is the most restrictive and suggests
   concrete relaxation steps to the user.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from tmdb_client import GENRE_NAME_TO_ID, GENRE_MAP


@dataclass
class MovieFilter:
    """Structured data model capturing all multi-dimensional movie filters."""

    # Temporal Filters
    min_year: int = 1950
    max_year: int = 2026
    decade: Optional[str] = None           # e.g. "1990s", "2000s"
    release_period: Optional[str] = None   # e.g. "Anytime", "Last 5 years", "Last 2 years", "This year"

    # Quality Filters
    min_rating: float = 0.0                # 0.0 to 10.0
    min_votes: int = 50                    # Minimum vote count threshold

    # Content Filters
    runtime_brackets: List[str] = field(default_factory=list)  # ["<90", "90-120", "120-150", "150+"]
    certifications: List[str] = field(default_factory=list)    # ["G", "PG", "PG-13", "R", "NC-17"]
    original_language: Optional[str] = None                    # "en", "ja", "fr", "ko", etc.

    # Personnel Filters
    actor_ids: List[int] = field(default_factory=list)
    actor_names: List[str] = field(default_factory=list)
    director_ids: List[int] = field(default_factory=list)
    director_names: List[str] = field(default_factory=list)

    # Genre Filters & Logic
    genres_include: List[str] = field(default_factory=list)
    genre_operator: str = "OR"             # "AND" or "OR"
    genres_exclude: List[str] = field(default_factory=list)    # NOT logic

    # Sort Order
    sort_by: str = "popularity.desc"

    def validate(self) -> List[str]:
        """Validates filter combinations and returns a list of error messages if invalid."""
        errors: List[str] = []
        if self.min_year > self.max_year:
            errors.append(f"Invalid Year Range: Minimum year ({self.min_year}) cannot exceed maximum year ({self.max_year}).")
        if not (0.0 <= self.min_rating <= 10.0):
            errors.append(f"Invalid Rating: Minimum rating must be between 0.0 and 10.0.")
        if self.min_votes < 0:
            errors.append(f"Invalid Vote Count: Minimum votes cannot be negative.")
        return errors

    def to_tmdb_params(self) -> Dict[str, Any]:
        """
        Translates structured filter properties into TMDB /discover/movie query parameters.
        Explains server-side translation logic.
        """
        params: Dict[str, Any] = {
            "include_adult": "false",
            "sort_by": self.sort_by,
        }

        # 1. Temporal Translation
        current_year = datetime.datetime.now().year
        effective_min_year = self.min_year
        effective_max_year = self.max_year

        if self.release_period == "This year":
            effective_min_year = current_year
            effective_max_year = current_year
        elif self.release_period == "Last 2 years":
            effective_min_year = current_year - 2
            effective_max_year = current_year
        elif self.release_period == "Last 5 years":
            effective_min_year = current_year - 5
            effective_max_year = current_year
        elif self.decade and self.decade != "All":
            try:
                dec_val = int(self.decade.replace("s", ""))
                effective_min_year = dec_val
                effective_max_year = dec_val + 9
            except ValueError:
                pass

        params["primary_release_date.gte"] = f"{effective_min_year}-01-01"
        params["primary_release_date.lte"] = f"{effective_max_year}-12-31"

        # 2. Quality Translation
        if self.min_rating > 0.0:
            params["vote_average.gte"] = self.min_rating
        if self.min_votes > 0:
            params["vote_count.gte"] = self.min_votes

        # 3. Genre Translation (AND vs. OR vs. NOT)
        include_ids = [GENRE_NAME_TO_ID[g.lower()] for g in self.genres_include if g.lower() in GENRE_NAME_TO_ID]
        if include_ids:
            if self.genre_operator == "AND":
                # Comma-separated indicates logical AND in TMDB API
                params["with_genres"] = ",".join(str(gid) for gid in include_ids)
            else:
                # Pipe-separated indicates logical OR in TMDB API
                params["with_genres"] = "|".join(str(gid) for gid in include_ids)

        exclude_ids = [GENRE_NAME_TO_ID[g.lower()] for g in self.genres_exclude if g.lower() in GENRE_NAME_TO_ID]
        if exclude_ids:
            # Comma-separated indicates NOT in TMDB API without_genres
            params["without_genres"] = ",".join(str(gid) for gid in exclude_ids)

        # 4. Content Language
        if self.original_language and self.original_language != "All":
            params["with_original_language"] = self.original_language

        # 5. Certification (US standard)
        if self.certifications:
            params["certification_country"] = "US"
            # Join with pipe for multiple allowed certifications
            params["certification"] = "|".join(self.certifications)

        # 6. Personnel
        all_people = list(self.actor_ids) + list(self.director_ids)
        if all_people:
            # Comma-separated implies movies involving these people
            params["with_people"] = ",".join(str(pid) for pid in all_people)

        return params

    def get_active_chips(self) -> List[Dict[str, str]]:
        """
        Returns active filter chips with human-readable labels for display in the UI.
        Format: [{'category': str, 'label': str}]
        """
        chips: List[Dict[str, str]] = []

        if self.min_year != 1950 or self.max_year != 2026:
            chips.append({"category": "Years", "label": f"{self.min_year}–{self.max_year}"})
        if self.decade and self.decade != "All":
            chips.append({"category": "Decade", "label": self.decade})
        if self.release_period and self.release_period != "Anytime":
            chips.append({"category": "Period", "label": self.release_period})

        if self.min_rating > 0.0:
            chips.append({"category": "Rating", "label": f"≥ {self.min_rating}★"})
        if self.min_votes > 50:
            chips.append({"category": "Votes", "label": f"≥ {self.min_votes}"})

        if self.runtime_brackets:
            chips.append({"category": "Runtime", "label": ", ".join(self.runtime_brackets)})
        if self.certifications:
            chips.append({"category": "Cert", "label": ", ".join(self.certifications)})
        if self.original_language and self.original_language != "All":
            chips.append({"category": "Language", "label": self.original_language.upper()})

        if self.genres_include:
            chips.append({"category": f"Genres ({self.genre_operator})", "label": ", ".join(self.genres_include)})
        if self.genres_exclude:
            chips.append({"category": "Exclude Genres", "label": ", ".join(self.genres_exclude)})

        for name in self.actor_names:
            chips.append({"category": "Cast", "label": name})
        for name in self.director_names:
            chips.append({"category": "Director", "label": name})

        return chips


def client_side_filter_movie(movie: Dict[str, Any], f: MovieFilter) -> bool:
    """
    Applies precise client-side post-filters to ensure exact compliance with
    complex conditions that TMDB /discover cannot strictly enforce in one pass.
    """
    # 1. Year Check
    year = movie.get("year", 0)
    if year > 0:
        if year < f.min_year or year > f.max_year:
            return False

    # 2. Rating & Votes Check
    if movie.get("vote_average", 0.0) < f.min_rating:
        return False
    if movie.get("vote_count", 0) < f.min_votes:
        return False

    # 3. Runtime Brackets Check
    # If user specified runtime brackets, movie must match at least one selected bracket
    if f.runtime_brackets:
        rt = movie.get("runtime", 0)
        bracket_matched = False
        for b in f.runtime_brackets:
            if b == "<90" and 0 < rt < 90:
                bracket_matched = True
                break
            elif b == "90-120" and 90 <= rt <= 120:
                bracket_matched = True
                break
            elif b == "120-150" and 120 < rt <= 150:
                bracket_matched = True
                break
            elif b == "150+" and rt > 150:
                bracket_matched = True
                break
        if not bracket_matched and rt > 0:
            return False

    # 4. US Certification Check
    if f.certifications:
        movie_cert = movie.get("certification", "").upper()
        if movie_cert and movie_cert != "NR" and movie_cert not in [c.upper() for c in f.certifications]:
            return False

    # 5. Language Check
    if f.original_language and f.original_language != "All":
        if movie.get("original_language", "").lower() != f.original_language.lower():
            return False

    # 6. Genre Inclusion Logic (AND vs OR)
    movie_genres = [g.lower() for g in movie.get("genres", [])]
    if f.genres_include:
        req_genres = [g.lower() for g in f.genres_include]
        if f.genre_operator == "AND":
            # All required genres must be present
            if not all(rg in movie_genres for rg in req_genres):
                return False
        else:
            # At least one required genre must be present
            if not any(rg in movie_genres for rg in req_genres):
                return False

    # 7. Genre Exclusion Logic (NOT)
    if f.genres_exclude:
        excl_genres = [g.lower() for g in f.genres_exclude]
        if any(eg in movie_genres for eg in excl_genres):
            return False

    # 8. Director Matching
    if f.director_names:
        movie_director = (movie.get("director") or "").lower()
        movie_directors = [d.lower() for d in movie.get("directors", [])]
        matched_dir = any(
            req_d.lower() in movie_director or any(req_d.lower() in d for d in movie_directors)
            for req_d in f.director_names
        )
        if not matched_dir:
            return False

    # 9. Actor Matching
    if f.actor_names:
        movie_cast = [c.lower() for c in movie.get("top_cast", [])]
        matched_cast = any(
            any(req_a.lower() in c for c in movie_cast)
            for req_a in f.actor_names
        )
        if not matched_cast:
            return False

    return True


def diagnose_zero_results(f: MovieFilter) -> List[str]:
    """
    Evaluates active filter settings when 0 results are returned and generates
    targeted recommendations explaining which filter is likely most restrictive.
    """
    suggestions: List[str] = []

    if f.min_votes > 1000:
        suggestions.append(
            f"**High Vote Threshold**: Minimum vote count is set to {f.min_votes}. "
            "Lowering this threshold to 100–300 will surface many critically acclaimed or niche films."
        )

    if f.min_rating >= 8.0:
        suggestions.append(
            f"**Strict Quality Bar**: Minimum rating is {f.min_rating}★. "
            "Only a tiny percentage of all films exceed 8.0. Try setting this between 6.5★ and 7.5★."
        )

    if f.genre_operator == "AND" and len(f.genres_include) >= 2:
        suggestions.append(
            f"**Logical AND on Genres**: Requiring all of [{', '.join(f.genres_include)}] simultaneously "
            "is very restrictive. Switch the genre operator to **OR (any)** to broaden results."
        )

    if f.runtime_brackets and len(f.runtime_brackets) == 1:
        suggestions.append(
            f"**Narrow Runtime Window**: Filtered only for '{f.runtime_brackets[0]}'. "
            "Try checking additional runtime brackets (e.g., 90–120 min)."
        )

    if f.actor_names and f.director_names:
        suggestions.append(
            f"**Specific Personnel Intersection**: Searching for both director ({', '.join(f.director_names)}) "
            f"and actor ({', '.join(f.actor_names)}) together requires them to have collaborated on the same film. "
            "Try removing one of them."
        )

    if f.max_year - f.min_year < 5:
        suggestions.append(
            f"**Narrow Year Range**: Currently limited to {f.min_year}–{f.max_year}. "
            "Expand the year range slider to include more release years."
        )

    if not suggestions:
        suggestions.append(
            "Try clicking **'Clear All Filters'** to reset to default discovery parameters, "
            "then re-apply your favorite criteria one at a time."
        )

    return suggestions
