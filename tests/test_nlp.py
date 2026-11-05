"""
test_nlp.py - Unit Tests for Rule-Based Query Parser, VADER Sentiment & TF-IDF Similarity

Tests:
1. Natural-language query parsing for genres, decades, runtime, actors, directors, mood, and ratings.
2. VADER sentiment scoring and mood classification.
3. TF-IDF vector computation and cosine similarity.
"""

import unittest
from nlp import (
    parse_natural_language_query,
    analyze_sentiment,
    classify_mood,
    compute_sentiment_match,
    compute_tfidf_matrix,
    compute_similarity,
    extract_top_contributing_terms,
)


class TestNLP(unittest.TestCase):

    def test_parse_query_matrix(self):
        """Tests parsing: 'dark sci-fi from the 90s under 2 hours with Keanu Reeves'"""
        query = "dark sci-fi from the 90s under 2 hours with Keanu Reeves"
        res = parse_natural_language_query(query)

        self.assertIn("Science Fiction", res["genres"])
        self.assertEqual(res["decade"], "1990s")
        self.assertEqual(res["min_year"], 1990)
        self.assertEqual(res["max_year"], 1999)
        self.assertEqual(res["max_runtime"], 120)
        self.assertEqual(res["mood"], "Dark")
        self.assertIn("Keanu Reeves", res["actors"])

    def test_parse_query_nolan(self):
        """Tests parsing: 'intense thrillers directed by Christopher Nolan rated above 8'"""
        query = "intense thrillers directed by Christopher Nolan rated above 8"
        res = parse_natural_language_query(query)

        self.assertIn("Thriller", res["genres"])
        self.assertEqual(res["mood"], "Intense")
        self.assertIn("Christopher Nolan", res["directors"])
        self.assertAlmostEqual(res["min_rating"], 8.0, places=1)

    def test_parse_query_animation(self):
        """Tests parsing: 'uplifting animated movies from the 2000s'"""
        query = "uplifting animated movies from the 2000s"
        res = parse_natural_language_query(query)

        self.assertIn("Animation", res["genres"])
        self.assertEqual(res["decade"], "2000s")
        self.assertEqual(res["mood"], "Uplifting")

    def test_sentiment_scoring(self):
        """Verifies that positive plots score high compound and grim plots score negative."""
        happy_plot = "A heartwarming tale of love, triumph, joy, and wonderful friendship."
        dark_plot = "A grim story of murder, serial killers, terror, brutal destruction, and evil."

        sent_happy = analyze_sentiment(happy_plot)
        sent_dark = analyze_sentiment(dark_plot)

        self.assertGreater(sent_happy["compound"], 0.4)
        self.assertLess(sent_dark["compound"], -0.3)
        self.assertEqual(classify_mood(sent_happy), "Uplifting")
        self.assertEqual(classify_mood(sent_dark), "Dark")

    def test_sentiment_match_score(self):
        """Checks match score calculation between sentiment scores and target moods."""
        dark_sent = {"compound": -0.85, "pos": 0.05, "neg": 0.45, "neu": 0.50}
        score_dark = compute_sentiment_match(dark_sent, "Dark")
        score_uplifting = compute_sentiment_match(dark_sent, "Uplifting")

        self.assertGreater(score_dark, 0.8)
        self.assertLess(score_uplifting, 0.4)

    def test_tfidf_and_similarity(self):
        """Verifies TF-IDF matrix generation and cosine similarity calculation."""
        docs = [
            "cyberpunk artificial intelligence virtual reality keanureeves",
            "space exploration black hole astronaut relativity",
            "cyberpunk dystopia virtual simulation keanureeves",
        ]
        vec, matrix = compute_tfidf_matrix(docs)
        sims = compute_similarity(vec, matrix, target_idx=0)

        # Doc 0 vs Doc 0 is 1.0
        self.assertAlmostEqual(sims[0], 1.0, places=2)
        # Doc 0 should be much more similar to Doc 2 than to Doc 1
        self.assertGreater(sims[2], sims[1])

    def test_extract_top_contributing_terms(self):
        """Tests extraction of shared driving terms between two texts."""
        doc1 = "christophernolan batman joker gotham chaos crime"
        doc2 = "batman gotham vigilante joker dark knight"
        vec, _ = compute_tfidf_matrix([doc1, doc2])
        top_terms = extract_top_contributing_terms(vec, doc1, doc2, top_n=3)

        self.assertTrue(any(t in ("batman", "joker", "gotham") for t in top_terms))


if __name__ == "__main__":
    unittest.main()
