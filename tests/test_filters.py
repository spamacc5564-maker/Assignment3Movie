"""
test_filters.py - Unit Tests for Filter Validation, TMDB Parameter Mapping & Multi-Condition Logic

Tests:
1. Input boundary validation (e.g. min_year > max_year, out-of-range ratings).
2. Translation to TMDB /discover/movie parameters:
   - Temporal dates
   - Minimum rating and vote counts
   - Logical genre operators: AND (comma), OR (pipe), NOT (without_genres)
   - Personnel and language mapping
3. Complex COMBINATION client-side filtering:
   - Testing Genre AND + Genre NOT + Decade + Runtime + Director + Min Rating in combination.
4. Zero-result diagnostic engine suggestions.
"""

import unittest
from filters import MovieFilter, client_side_filter_movie, diagnose_zero_results
from tmdb_client import TMDBClient


class TestFilters(unittest.TestCase):

    def setUp(self):
        self.mock_movies = TMDBClient.get_mock_movies()

    def test_filter_validation(self):
        """Checks validation boundaries for years, ratings, and votes."""
        # Valid filter
        valid_f = MovieFilter(min_year=1990, max_year=2000, min_rating=7.0, min_votes=100)
        self.assertEqual(len(valid_f.validate()), 0)

        # Invalid year range (min_year > max_year)
        invalid_f1 = MovieFilter(min_year=2020, max_year=1990)
        errors = invalid_f1.validate()
        self.assertTrue(any("cannot exceed" in e for e in errors))

        # Invalid rating
        invalid_f2 = MovieFilter(min_rating=11.5)
        errors = invalid_f2.validate()
        self.assertTrue(any("between 0.0 and 10.0" in e for e in errors))

    def test_tmdb_param_mapping_temporal(self):
        """Verifies mapping of years and decades to TMDB release dates."""
        f_decade = MovieFilter(decade="1990s")
        params = f_decade.to_tmdb_params()
        self.assertEqual(params["primary_release_date.gte"], "1990-01-01")
        self.assertEqual(params["primary_release_date.lte"], "1999-12-31")

        f_range = MovieFilter(min_year=1975, max_year=1985)
        params_range = f_range.to_tmdb_params()
        self.assertEqual(params_range["primary_release_date.gte"], "1975-01-01")
        self.assertEqual(params_range["primary_release_date.lte"], "1985-12-31")

    def test_tmdb_param_mapping_genres(self):
        """Verifies AND vs. OR vs. NOT logical mapping for genre IDs."""
        # OR logic should produce pipe-separated IDs
        f_or = MovieFilter(genres_include=["Action", "Science Fiction"], genre_operator="OR")
        params_or = f_or.to_tmdb_params()
        self.assertIn("|", params_or["with_genres"])
        self.assertNotIn(",", params_or["with_genres"])

        # AND logic should produce comma-separated IDs
        f_and = MovieFilter(genres_include=["Action", "Science Fiction"], genre_operator="AND")
        params_and = f_and.to_tmdb_params()
        self.assertIn(",", params_and["with_genres"])
        self.assertNotIn("|", params_and["with_genres"])

        # NOT logic should populate without_genres
        f_not = MovieFilter(genres_exclude=["Horror", "Romance"])
        params_not = f_not.to_tmdb_params()
        self.assertIn("without_genres", params_not)

    def test_filters_in_combination(self):
        """
        Tests filters in COMBINATION:
        - Genre AND: Action AND Science Fiction
        - Genre NOT: Crime (must exclude movies with Crime)
        - Decade: 1990s
        - Runtime: 120-150 min
        - Director: Lana Wachowski
        - Min rating: 8.0
        Must match 'The Matrix' (1999) and reject others.
        """
        f = MovieFilter(
            min_year=1990,
            max_year=1999,
            min_rating=8.0,
            min_votes=1000,
            runtime_brackets=["120-150"],
            genres_include=["Action", "Science Fiction"],
            genre_operator="AND",
            genres_exclude=["Crime", "Horror"],
            director_names=["Lana Wachowski"]
        )

        matched = [m for m in self.mock_movies if client_side_filter_movie(m, f)]

        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["title"], "The Matrix")

    def test_filters_combination_rejection(self):
        """Tests that a conflicting filter correctly eliminates candidate films."""
        # Inception is 2010, Sci-Fi/Action, directed by Christopher Nolan, runtime 148 min.
        # If we require runtime '<90', it must be rejected.
        f = MovieFilter(
            min_year=2000,
            max_year=2020,
            runtime_brackets=["<90"],
            director_names=["Christopher Nolan"]
        )

        matched = [m for m in self.mock_movies if client_side_filter_movie(m, f)]
        titles = [m["title"] for m in matched]
        self.assertNotIn("Inception", titles)
        self.assertNotIn("Interstellar", titles)

    def test_diagnose_zero_results(self):
        """Ensures that diagnostic messages are informative and actionable."""
        f_restrictive = MovieFilter(
            min_votes=3000,
            min_rating=9.2,
            genre_operator="AND",
            genres_include=["Action", "Drama", "Animation", "Horror"]
        )
        suggestions = diagnose_zero_results(f_restrictive)
        self.assertTrue(len(suggestions) >= 2)
        combined_text = " ".join(suggestions)
        self.assertTrue("vote count" in combined_text.lower() or "rating" in combined_text.lower())


if __name__ == "__main__":
    unittest.main()
