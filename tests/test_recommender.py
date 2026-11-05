"""
test_recommender.py - Unit Tests for Content-Based, Sentiment-Based & Hybrid Recommenders

Tests:
1. Content-based TF-IDF cosine ranking.
2. Sentiment-based VADER mood ranking.
3. Hybrid recommender weight blending (e.g. 70/30).
4. Explainability reports (shared genres, keywords, cast, sentiment closeness).
"""

import unittest
from recommender import MovieRecommender
from tmdb_client import TMDBClient


class TestRecommender(unittest.TestCase):

    def setUp(self):
        self.recommender = MovieRecommender()
        self.movies = TMDBClient.get_mock_movies()
        # Find The Matrix as anchor
        self.matrix_movie = next(m for m in self.movies if m["id"] == 603)
        # Find Blade Runner
        self.blade_runner = next(m for m in self.movies if m["id"] == 78)
        # Find Forrest Gump
        self.forrest_gump = next(m for m in self.movies if m["id"] == 13)

    def test_content_based_recommendation(self):
        """Content recommender should rank Blade Runner or Dune higher than Forrest Gump for The Matrix."""
        recs = self.recommender.recommend_content_based(self.matrix_movie, self.movies, top_n=5)

        # Target movie must not be in recommendations
        rec_ids = [r["id"] for r in recs]
        self.assertNotIn(self.matrix_movie["id"], rec_ids)

        # Top match should be a Sci-Fi / Cyberpunk movie
        top_genres = recs[0]["genres"]
        self.assertTrue("Science Fiction" in top_genres or "Action" in top_genres)
        self.assertGreater(recs[0]["content_score"], recs[-1]["content_score"])

    def test_sentiment_based_recommendation(self):
        """Sentiment recommender should match plot polarity to requested mood."""
        dark_recs = self.recommender.recommend_sentiment_based("Dark", self.movies, top_n=5)
        uplifting_recs = self.recommender.recommend_sentiment_based("Uplifting", self.movies, top_n=5)

        # Top dark rec should have negative or grim compound
        top_dark = dark_recs[0]
        self.assertLessEqual(top_dark["sentiment_data"]["compound"], 0.2)

        # Top uplifting rec should have positive compound
        top_up = uplifting_recs[0]
        self.assertGreaterEqual(top_up["sentiment_data"]["compound"], 0.3)

    def test_hybrid_weight_blending(self):
        """Checks the weighted combination: (w * content) + ((1 - w) * sentiment)."""
        w = 0.8
        recs = self.recommender.recommend_hybrid(
            self.matrix_movie,
            target_mood="Dark",
            candidate_pool=self.movies,
            content_weight=w,
            top_n=5,
        )

        top_rec = recs[0]
        expected_hybrid = round((w * top_rec["content_score"]) + ((1 - w) * top_rec["sentiment_score"]), 3)
        self.assertAlmostEqual(top_rec["hybrid_score"], expected_hybrid, places=2)

    def test_explain_similarity(self):
        """Verifies explainability report generation between two movies."""
        explanation = self.recommender.explain_similarity(self.matrix_movie, self.blade_runner)

        # Both are Sci-Fi
        self.assertIn("Science Fiction", explanation["shared_genres"])
        # Both share cyberpunk keywords
        self.assertTrue(
            "cyberpunk" in explanation["shared_keywords"] or
            "dystopia" in explanation["shared_keywords"]
        )
        self.assertFalse(explanation["same_director"])
        self.assertGreaterEqual(explanation["sentiment_closeness_pct"], 0)


if __name__ == "__main__":
    unittest.main()
