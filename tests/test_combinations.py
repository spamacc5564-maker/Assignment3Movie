"""
test_combinations.py - Deep Verification of Complex Filter Combinations

Verifies:
1. Genre AND + NOT + Decade + Runtime bracket + Director + Min rating.
2. Multiple Actors + Decades + Certification + Language.
3. Relaxing constraints sequentially when zero results are found.
"""

import unittest
from filters import MovieFilter, client_side_filter_movie, diagnose_zero_results
from tmdb_client import TMDBClient


class TestComplexFilterCombinations(unittest.TestCase):

    def setUp(self):
        self.movies = TMDBClient.get_mock_movies()

    def test_nolan_interstellar_combination(self):
        """
        Combination test for Interstellar (2014):
        - Genres: Adventure AND Science Fiction
        - Exclude: Horror
        - Decade: 2010s (min_year 2010, max_year 2019)
        - Runtime: 150+ min (Interstellar is 169 min)
        - Director: Christopher Nolan
        - Actor: Matthew McConaughey
        - Min Rating: 8.0
        - Certification: PG-13
        """
        f = MovieFilter(
            min_year=2010,
            max_year=2019,
            min_rating=8.0,
            min_votes=1000,
            runtime_brackets=["150+"],
            certifications=["PG-13"],
            original_language="en",
            genres_include=["Adventure", "Science Fiction"],
            genre_operator="AND",
            genres_exclude=["Horror"],
            director_names=["Christopher Nolan"],
            actor_names=["Matthew McConaughey"]
        )

        matched = [m for m in self.movies if client_side_filter_movie(m, f)]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["title"], "Interstellar")

    def test_tarantino_pulp_fiction_combination(self):
        """
        Combination test for Pulp Fiction (1994):
        - Genres: Thriller OR Crime
        - Exclude: Science Fiction
        - Decade: 1990s
        - Runtime: 150+ min (Pulp Fiction is 154 min)
        - Director: Quentin Tarantino
        - Min Rating: 8.0
        - Certification: R
        """
        f = MovieFilter(
            min_year=1990,
            max_year=1999,
            min_rating=8.0,
            min_votes=500,
            runtime_brackets=["150+"],
            certifications=["R"],
            genres_include=["Thriller", "Crime"],
            genre_operator="OR",
            genres_exclude=["Science Fiction"],
            director_names=["Quentin Tarantino"],
        )

        matched = [m for m in self.movies if client_side_filter_movie(m, f)]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["title"], "Pulp Fiction")

    def test_japanese_animation_combination(self):
        """
        Combination test for Spirited Away (2001):
        - Genres: Animation AND Fantasy
        - Language: ja
        - Decade: 2000s
        - Certification: PG
        - Min Rating: 8.0
        """
        f = MovieFilter(
            min_year=2000,
            max_year=2009,
            min_rating=8.0,
            original_language="ja",
            certifications=["PG"],
            genres_include=["Animation", "Fantasy"],
            genre_operator="AND",
        )

        matched = [m for m in self.movies if client_side_filter_movie(m, f)]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["title"], "Spirited Away")

    def test_overconstrained_zero_result_relaxation(self):
        """Tests that when an impossible combination produces 0 results, diagnostics advise relaxation."""
        f_impossible = MovieFilter(
            min_year=1990,
            max_year=1999,
            min_rating=9.5,  # Unrealistic rating
            genres_include=["Animation", "Western"],
            genre_operator="AND",
            min_votes=4000,
        )

        matched = [m for m in self.movies if client_side_filter_movie(m, f_impossible)]
        self.assertEqual(len(matched), 0)

        suggestions = diagnose_zero_results(f_impossible)
        self.assertTrue(len(suggestions) > 0)


if __name__ == "__main__":
    unittest.main()
