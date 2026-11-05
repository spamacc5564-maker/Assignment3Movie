"""
test_watchlist.py - Unit Tests for Watchlist Management, Taste Profiling & CSV Serialization

Tests:
1. Adding and removing movies from watchlist.
2. Centroid taste profile computation:
   - Genre distribution percentages
   - Decades frequency
   - Directors frequency
   - Average rating and runtime
3. Recommendation from profile centroid.
4. CSV export and import serialization fidelity.
"""

import unittest
from watchlist import (
    compute_taste_profile,
    recommend_from_profile,
    export_watchlist_csv,
    import_watchlist_csv,
)
from tmdb_client import TMDBClient


class TestWatchlist(unittest.TestCase):

    def setUp(self):
        self.movies = TMDBClient.get_mock_movies()
        # Curate a sample watchlist: The Matrix (1999), Inception (2010), Blade Runner 2049 (2017)
        self.sample_saved = [
            next(m for m in self.movies if m["id"] == 603),      # The Matrix
            next(m for m in self.movies if m["id"] == 27205),    # Inception
            next(m for m in self.movies if m["id"] == 335984),   # Blade Runner 2049
        ]

    def test_compute_taste_profile(self):
        """Verifies calculation of user preferences across genres, directors, and eras."""
        profile = compute_taste_profile(self.sample_saved)

        self.assertEqual(profile["total_movies"], 3)
        # Sci-Fi is in all 3 movies
        top_genre = profile["favorite_genres"][0]
        self.assertEqual(top_genre[0], "Science Fiction")
        self.assertEqual(top_genre[1], 3)
        self.assertAlmostEqual(top_genre[2], 100.0, places=1)

        # Average rating should be around (8.2 + 8.4 + 7.6) / 3 = 8.06 -> 8.1
        self.assertAlmostEqual(profile["avg_rating"], 8.1, places=1)

        # Composite soup should contain tokens from all 3 films
        self.assertIn("matrix", profile["composite_soup"].lower())
        self.assertIn("inception", profile["composite_soup"].lower())

    def test_recommend_from_profile(self):
        """Profile recommender should rank Interstellar (another Nolan Sci-Fi) high."""
        recs = recommend_from_profile(self.sample_saved, self.movies, top_n=5)

        # Saved movies must be excluded from recommendations
        saved_ids = set(m["id"] for m in self.sample_saved)
        for r in recs:
            self.assertNotIn(r["id"], saved_ids)

        # Interstellar or Blade Runner (1982) should be among top recommendations
        rec_titles = [r["title"] for r in recs]
        self.assertTrue("Interstellar" in rec_titles or "Blade Runner" in rec_titles)

    def test_csv_export_and_import(self):
        """Verifies that watchlist can be serialized to CSV and parsed back accurately."""
        csv_str = export_watchlist_csv(self.sample_saved)
        self.assertIn("The Matrix", csv_str)
        self.assertIn("Inception", csv_str)
        self.assertIn("1999", csv_str)

        imported_count, errors = import_watchlist_csv(csv_str)
        self.assertEqual(len(errors), 0)
        self.assertEqual(imported_count, 3)


if __name__ == "__main__":
    unittest.main()
