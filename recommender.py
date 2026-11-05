"""
recommender.py - Content-Based, Sentiment-Based, and Hybrid Recommendation Engines

Algorithmic Design & Decisions:
1. Content-Based Recommendation:
   - Uses TF-IDF representation of weighted metadata 'soup' (director, cast, genres, keywords, plot).
   - Computes cosine similarity between target movie / user profile vector and candidate pool.
   - Values range from 0.0 to 1.0 (cosine angle between non-negative vectors).

2. Sentiment-Based Recommendation:
   - Evaluates VADER polarity profiles of plot overviews against target mood prototypes:
     * Uplifting: Target compound ~ +0.80, low negativity.
     * Dark: Target compound ~ -0.80, elevated negativity.
     * Intense: High polarity magnitude |compound| >= 0.65.
     * Light-hearted: Moderate positivity (+0.35) with minimal negative words.
     * Neutral: Objective, factual tone (compound near 0.0).
   - Distance between candidate sentiment and target archetype yields a normalized [0, 1] fit.

3. Hybrid Recommendation:
   - Linear convex combination:
     Score_hybrid = (w_content * Score_content) + ((1 - w_content) * Score_sentiment)
   - Default: w_content = 0.70 (70% content, 30% sentiment).
   - Allows users to adjust the balance: prioritizing thematic/personnel continuity (high content)
     versus emotional resonance / atmospheric vibe (high sentiment).

4. Multi-Dimensional Explainability Engine:
   Breaks down recommendation rationale into 5 transparent signals:
   - Shared Genres
   - Shared Keywords
   - Shared Cast & Crew
   - Top Contributing TF-IDF tokens
   - Sentiment Profile Alignment
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from features import build_movie_soup, enrich_movie_with_features
from nlp import (
    analyze_sentiment,
    classify_mood,
    compute_sentiment_match,
    compute_tfidf_matrix,
    compute_similarity,
    compute_vector_similarity,
    extract_top_contributing_terms,
)

logger = logging.getLogger(__name__)


class MovieRecommender:
    """
    Unified Movie Recommendation Engine providing Content, Sentiment, and Hybrid modes.
    """

    def __init__(self):
        pass

    def recommend_content_based(
        self,
        target_movie: Dict[str, Any],
        candidate_pool: List[Dict[str, Any]],
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Recommends top N movies from candidate_pool most similar to target_movie
        based on TF-IDF cosine similarity over metadata soup.
        """
        if not candidate_pool:
            return []

        # Filter out exact duplicate of target movie
        pool = [m for m in candidate_pool if m.get("id") != target_movie.get("id")]
        if not pool:
            return []

        target_soup = build_movie_soup(target_movie)
        pool_soups = [build_movie_soup(m) for m in pool]

        all_documents = [target_soup] + pool_soups
        vectorizer, matrix = compute_tfidf_matrix(all_documents)

        # Target is index 0; candidates are indices 1..len(pool)
        sim_scores = compute_similarity(vectorizer, matrix, target_idx=0)
        candidate_scores = sim_scores[1:]

        scored_results: List[Dict[str, Any]] = []
        for i, m in enumerate(pool):
            sim = candidate_scores[i]
            rec = dict(m)
            rec["content_score"] = round(sim, 3)
            rec["sentiment_score"] = 0.0
            rec["match_score"] = round(sim * 100)
            rec["match_pct"] = f"{round(sim * 100)}%"
            scored_results.append(rec)

        scored_results.sort(key=lambda x: x["content_score"], reverse=True)
        return scored_results[:top_n]

    def recommend_sentiment_based(
        self,
        target_mood: str,
        candidate_pool: List[Dict[str, Any]],
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Recommends top N movies from candidate_pool matching the specified emotional mood
        using VADER sentiment polarity analysis on plot overviews.
        """
        if not candidate_pool:
            return []

        scored_results: List[Dict[str, Any]] = []
        for m in candidate_pool:
            overview = m.get("overview", "")
            sent = analyze_sentiment(overview)
            fit = compute_sentiment_match(sent, target_mood)

            rec = dict(m)
            rec["sentiment_data"] = sent
            rec["detected_mood"] = classify_mood(sent)
            rec["sentiment_score"] = round(fit, 3)
            rec["content_score"] = 0.0
            rec["match_score"] = round(fit * 100)
            rec["match_pct"] = f"{round(fit * 100)}%"
            scored_results.append(rec)

        scored_results.sort(key=lambda x: x["sentiment_score"], reverse=True)
        return scored_results[:top_n]

    def recommend_hybrid(
        self,
        target_movie: Optional[Dict[str, Any]],
        target_mood: str,
        candidate_pool: List[Dict[str, Any]],
        content_weight: float = 0.7,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Blends Content-Based and Sentiment-Based recommendations with a user-adjustable slider weight.
        Score_hybrid = (content_weight * Score_content) + ((1 - content_weight) * Score_sentiment)
        """
        if not candidate_pool:
            return []

        # If no target movie is given, fallback purely to sentiment
        if target_movie is None:
            return self.recommend_sentiment_based(target_mood, candidate_pool, top_n=top_n)

        # Filter out target movie from candidates
        pool = [m for m in candidate_pool if m.get("id") != target_movie.get("id")]
        if not pool:
            return []

        # 1. Compute Content Similarities
        target_soup = build_movie_soup(target_movie)
        pool_soups = [build_movie_soup(m) for m in pool]
        all_documents = [target_soup] + pool_soups

        vectorizer, matrix = compute_tfidf_matrix(all_documents)
        sim_scores = compute_similarity(vectorizer, matrix, target_idx=0)
        content_scores = sim_scores[1:]

        # 2. Compute Sentiment Similarities
        w_c = max(0.0, min(1.0, content_weight))
        w_s = 1.0 - w_c

        scored_results: List[Dict[str, Any]] = []
        for i, m in enumerate(pool):
            c_score = content_scores[i]

            sent = analyze_sentiment(m.get("overview", ""))
            s_score = compute_sentiment_match(sent, target_mood)

            hybrid_score = (w_c * c_score) + (w_s * s_score)

            rec = dict(m)
            rec["content_score"] = round(c_score, 3)
            rec["sentiment_score"] = round(s_score, 3)
            rec["sentiment_data"] = sent
            rec["detected_mood"] = classify_mood(sent)
            rec["hybrid_score"] = round(hybrid_score, 3)
            rec["weights_used"] = f"{int(w_c * 100)}% Content / {int(w_s * 100)}% Mood ({target_mood})"
            rec["match_score"] = round(hybrid_score * 100)
            rec["match_pct"] = f"{round(hybrid_score * 100)}%"
            scored_results.append(rec)

        scored_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return scored_results[:top_n]

    def explain_similarity(
        self,
        target_movie: Dict[str, Any],
        candidate_movie: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Produces a rich explainability report comparing two movies across genres, keywords,
        cast/director, top TF-IDF features, and emotional tone.
        """
        # Shared Genres
        t_genres = set(g.lower() for g in target_movie.get("genres", []))
        c_genres = set(g.lower() for g in candidate_movie.get("genres", []))
        shared_genres = [g.title() for g in t_genres.intersection(c_genres)]

        # Shared Keywords
        t_kw = set(k.lower() for k in target_movie.get("keywords", []))
        c_kw = set(k.lower() for k in candidate_movie.get("keywords", []))
        shared_keywords = list(t_kw.intersection(c_kw))

        # Shared Cast
        t_cast = set(c.lower() for c in target_movie.get("top_cast", []))
        c_cast = set(c.lower() for c in candidate_movie.get("top_cast", []))
        shared_cast = [c.title() for c in t_cast.intersection(c_cast)]

        # Director match
        t_dir = (target_movie.get("director") or "").strip()
        c_dir = (candidate_movie.get("director") or "").strip()
        same_director = (bool(t_dir) and t_dir != "Unknown" and t_dir.lower() == c_dir.lower())

        # Top TF-IDF driving terms
        t_soup = build_movie_soup(target_movie)
        c_soup = build_movie_soup(candidate_movie)
        vectorizer, _ = compute_tfidf_matrix([t_soup, c_soup])
        top_terms = extract_top_contributing_terms(vectorizer, t_soup, c_soup, top_n=5)

        # Sentiment alignment
        t_sent = analyze_sentiment(target_movie.get("overview", ""))
        c_sent = analyze_sentiment(candidate_movie.get("overview", ""))
        compound_diff = abs(t_sent["compound"] - c_sent["compound"])
        sentiment_closeness = max(0.0, min(1.0, 1.0 - (compound_diff / 2.0)))

        t_mood = classify_mood(t_sent)
        c_mood = classify_mood(c_sent)

        if compound_diff <= 0.3:
            sentiment_summary = f"Strong emotional alignment: both films share a {t_mood.lower()} tone."
        else:
            sentiment_summary = f"Contrasting emotional vibe: '{target_movie.get('title')}' is {t_mood.lower()} while '{candidate_movie.get('title')}' is {c_mood.lower()}."

        return {
            "shared_genres": shared_genres,
            "shared_keywords": shared_keywords,
            "shared_cast": shared_cast,
            "same_director": same_director,
            "director_name": t_dir if same_director else None,
            "top_tfidf_terms": top_terms,
            "target_sentiment": t_sent,
            "candidate_sentiment": c_sent,
            "target_mood": t_mood,
            "candidate_mood": c_mood,
            "sentiment_closeness_pct": round(sentiment_closeness * 100),
            "sentiment_summary": sentiment_summary,
        }
