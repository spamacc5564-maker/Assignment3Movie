"""
nlp.py - NLP Engine: TF-IDF Vectorization, VADER Sentiment, and NL Query Parser

Key Decisions & Rationale:
1. Why TF-IDF & Cosine Similarity?
   TF-IDF (Term Frequency - Inverse Document Frequency) downweights ubiquitous, uninformative
   tokens while amplifying distinctive niche words (e.g., 'cyberpunk', 'christophernolan').
   Cosine similarity measures angle rather than magnitude, ensuring long synopses do not
   unfairly score higher than concise, punchy loglines.

2. Why VADER (Valence Aware Dictionary and sEntiment Reasoner)?
   VADER is specifically calibrated for short-form narrative texts. It handles grammatical
   modifiers (e.g., 'not exciting', 'extremely grim', contrastive 'but') and computes
   four normalized outputs: compound (-1.0 to +1.0), pos (0 to 1), neg (0 to 1), and neu (0 to 1).
   This allows nuanced mood mapping (e.g., 'dark' matches high negative/low compound; 'uplifting'
   matches high positive/high compound; 'intense' matches high absolute compound).

3. Rule-Based Natural Language Query Parser:
   Instead of relying on fragile, paid LLM APIs or heavy external binaries, CineMatch uses
   a deterministic rule-based parser based on regular expressions, entity prefix extractors,
   and domain synonym dictionaries. It parses decades, runtime ceilings, actors, directors,
   ratings, genres, and mood profiles.

4. Explainable AI:
   Computes the element-wise Hadamard product of TF-IDF vectors (v_target * v_candidate)
   to extract the top 3-5 feature terms that actually drove the recommendation.
"""

from __future__ import annotations

import re
import math
import logging
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import Counter

logger = logging.getLogger(__name__)

# --- NLTK VADER Sentiment Initialization with Safe Fallback ---
_VADER_ANALYZER = None

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        nltk.download("vader_lexicon", quiet=True)
        _VADER_ANALYZER = SentimentIntensityAnalyzer()
    except Exception as e:
        logger.warning("Could not initialize NLTK VADER: %s. Using internal lexicon fallback.", e)
except ImportError:
    _VADER_ANALYZER = None


# Built-in lightweight lexicon fallback if NLTK is not present
EMOTION_LEXICON: Dict[str, float] = {
    # Positive / Uplifting
    "love": 3.0, "hope": 2.5, "hero": 2.2, "triumph": 3.2, "joy": 2.8, "wonderful": 3.0,
    "inspiration": 3.0, "courage": 2.5, "beautiful": 2.5, "humor": 2.0, "heartwarming": 3.0,
    "miracle": 3.2, "laughter": 2.4, "victory": 2.8, "friendship": 2.2, "adventure": 1.5,
    "save": 2.0, "freedom": 2.5, "magic": 2.0, "legendary": 2.2,
    # Negative / Dark / Grim
    "murder": -3.2, "kill": -3.0, "death": -2.8, "killer": -3.1, "serial": -2.0,
    "blood": -2.5, "grim": -2.5, "terror": -3.2, "dystopia": -2.4, "corrupt": -2.6,
    "revenge": -2.2, "chaos": -2.0, "deadly": -2.9, "threat": -2.2, "dark": -1.8,
    "nightmare": -3.0, "tragic": -2.8, "crime": -2.0, "horror": -3.0, "destruction": -2.8,
    "evil": -3.0, "sinister": -2.8, "fear": -2.4, "brutal": -3.1, "violence": -2.9,
    # Intensity boosters
    "action": 1.5, "explosion": 1.8, "chase": 1.4, "war": -2.2, "battle": 1.2,
}


def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyzes sentiment of text and returns a dict with:
      {'compound': float (-1.0 to 1.0), 'pos': float, 'neg': float, 'neu': float}
    """
    if not text:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

    # Use NLTK VADER if available
    if _VADER_ANALYZER is not None:
        try:
            return _VADER_ANALYZER.polarity_scores(text)
        except Exception:
            pass

    # Built-in lightweight fallback scoring
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

    total_score = 0.0
    pos_count = 0
    neg_count = 0

    for word in tokens:
        val = EMOTION_LEXICON.get(word, 0.0)
        if val > 0:
            pos_count += 1
            total_score += val
        elif val < 0:
            neg_count += 1
            total_score += val

    # Normalization formula matching VADER's compound: score / sqrt(score^2 + 15)
    compound = total_score / math.sqrt((total_score ** 2) + 15.0)
    pos = min(round(pos_count / len(tokens), 3), 1.0)
    neg = min(round(neg_count / len(tokens), 3), 1.0)
    neu = max(0.0, round(1.0 - (pos + neg), 3))

    return {
        "compound": round(compound, 3),
        "pos": pos,
        "neg": neg,
        "neu": neu,
    }


def classify_mood(scores: Dict[str, float]) -> str:
    """Classifies a sentiment score dictionary into a recognizable mood archetype."""
    compound = scores.get("compound", 0.0)
    neg = scores.get("neg", 0.0)
    pos = scores.get("pos", 0.0)

    if compound >= 0.50:
        return "Uplifting"
    elif compound <= -0.40:
        return "Dark"
    elif abs(compound) >= 0.65 or (neg > 0.15 and pos > 0.15):
        return "Intense"
    elif 0.15 <= compound < 0.50:
        return "Light-hearted"
    else:
        return "Neutral"


def compute_sentiment_match(scores: Dict[str, float], target_mood: str) -> float:
    """
    Computes a match score between 0.0 and 1.0 comparing movie sentiment scores
    against a desired target mood profile.
    """
    compound = scores.get("compound", 0.0)
    mood = target_mood.strip().lower()

    if mood in ("uplifting", "positive", "cheerful"):
        # Optimal compound is +0.8
        target = 0.8
        diff = abs(compound - target)
        return max(0.0, min(1.0, 1.0 - (diff / 1.8)))

    elif mood in ("dark", "grim", "somber", "gloomy"):
        # Optimal compound is -0.8
        target = -0.8
        diff = abs(compound - target)
        return max(0.0, min(1.0, 1.0 - (diff / 1.8)))

    elif mood in ("intense", "extreme", "action-packed"):
        # Intense means high polarity divergence
        magnitude = abs(compound)
        return max(0.0, min(1.0, magnitude))

    elif mood in ("light-hearted", "lighthearted", "fun", "breezy"):
        # Optimal compound is around +0.35 with low negative
        target = 0.35
        diff = abs(compound - target) + (scores.get("neg", 0.0) * 1.5)
        return max(0.0, min(1.0, 1.0 - (diff / 1.5)))

    elif mood in ("neutral", "balanced", "objective"):
        # Compound near 0.0
        diff = abs(compound)
        return max(0.0, min(1.0, 1.0 - (diff / 1.0)))

    return 0.5


# --- TF-IDF Vectorization & Cosine Similarity ---
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


class PurePythonTfidf:
    """
    Robust pure-Python TF-IDF and Cosine Similarity implementation.
    Acts as a 100% reliable fallback if scikit-learn is not available in the test environment.
    """

    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: Dict[str, float] = {}

    def fit_transform(self, documents: List[str]) -> List[Dict[int, float]]:
        doc_count = len(documents)
        if doc_count == 0:
            return []

        doc_terms = [re.findall(r"\b\w{2,}\b", doc.lower()) for doc in documents]
        df = Counter()
        for terms in doc_terms:
            df.update(set(terms))

        # Select top features
        most_common = [term for term, _ in df.most_common(self.max_features)]
        self.vocabulary_ = {term: idx for idx, term in enumerate(most_common)}

        # Compute smooth IDF: log((1 + N) / (1 + df)) + 1
        for term, idx in self.vocabulary_.items():
            self.idf_[term] = math.log((1 + doc_count) / (1 + df[term])) + 1.0

        # Build sparse L2-normalized document vectors
        vectors = []
        for terms in doc_terms:
            tf = Counter(terms)
            vec: Dict[int, float] = {}
            norm_sq = 0.0
            for term, count in tf.items():
                if term in self.vocabulary_:
                    idx = self.vocabulary_[term]
                    weight = count * self.idf_[term]
                    vec[idx] = weight
                    norm_sq += weight * weight
            norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
            for idx in vec:
                vec[idx] /= norm
            vectors.append(vec)
        return vectors

    def transform(self, documents: List[str]) -> List[Dict[int, float]]:
        vectors = []
        for doc in documents:
            terms = re.findall(r"\b\w{2,}\b", doc.lower())
            tf = Counter(terms)
            vec: Dict[int, float] = {}
            norm_sq = 0.0
            for term, count in tf.items():
                if term in self.vocabulary_:
                    idx = self.vocabulary_[term]
                    weight = count * self.idf_[term]
                    vec[idx] = weight
                    norm_sq += weight * weight
            norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
            for idx in vec:
                vec[idx] /= norm
            vectors.append(vec)
        return vectors


def compute_tfidf_matrix(documents: List[str]) -> Tuple[Any, Any]:
    """
    Computes the TF-IDF matrix for a list of document strings.
    Returns (vectorizer, tfidf_matrix).
    """
    if _SKLEARN_AVAILABLE:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(documents)
        return vectorizer, tfidf_matrix
    else:
        vectorizer = PurePythonTfidf(max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(documents)
        return vectorizer, tfidf_matrix


def compute_similarity(vectorizer: Any, matrix: Any, target_idx: int) -> List[float]:
    """
    Computes pairwise cosine similarity between a target row and all other rows.
    Returns a list of float similarity scores in [0.0, 1.0].
    """
    if _SKLEARN_AVAILABLE:
        target_vec = matrix[target_idx]
        sims = cosine_similarity(target_vec, matrix).flatten()
        return [float(s) for s in sims]
    else:
        target_vec = matrix[target_idx]
        sims = []
        for row in matrix:
            # Dot product of normalized sparse vectors
            dot = sum(target_vec.get(idx, 0.0) * val for idx, val in row.items())
            sims.append(max(0.0, min(1.0, dot)))
        return sims


def compute_vector_similarity(target_vec: Any, candidate_vecs: Any) -> List[float]:
    """
    Computes cosine similarity between an arbitrary query/profile vector and candidates.
    """
    if _SKLEARN_AVAILABLE:
        sims = cosine_similarity(target_vec, candidate_vecs).flatten()
        return [float(s) for s in sims]
    else:
        sims = []
        for row in candidate_vecs:
            dot = sum(target_vec.get(idx, 0.0) * val for idx, val in row.items())
            sims.append(max(0.0, min(1.0, dot)))
        return sims


def extract_top_contributing_terms(
    vectorizer: Any,
    doc_a: str,
    doc_b: str,
    top_n: int = 5
) -> List[str]:
    """
    Identifies the top N terms that drive similarity between two documents
    using the element-wise product of their TF-IDF representations.
    """
    if _SKLEARN_AVAILABLE:
        try:
            tfidf = vectorizer.transform([doc_a, doc_b])
            feature_names = vectorizer.get_feature_names_out()
            row_a = tfidf[0].toarray()[0]
            row_b = tfidf[1].toarray()[0]
            product = row_a * row_b
            top_indices = product.argsort()[::-1][:top_n]
            return [str(feature_names[i]) for i in top_indices if product[i] > 0]
        except Exception:
            pass

    # Pure Python fallback
    tokens_a = set(re.findall(r"\b\w{3,}\b", doc_a.lower()))
    tokens_b = set(re.findall(r"\b\w{3,}\b", doc_b.lower()))
    common = list(tokens_a.intersection(tokens_b))
    # Exclude common stop words
    stops = {"the", "and", "with", "from", "that", "this", "for", "was", "are", "his", "her", "their"}
    filtered = [w for w in common if w not in stops]
    return filtered[:top_n]


# --- Natural-Language Query Parser ---

def parse_natural_language_query(query: str) -> Dict[str, Any]:
    """
    Rule-based parser that maps free-text movie queries to structured CineMatch filters.

    Examples:
      - "dark sci-fi from the 90s under 2 hours with Keanu Reeves"
      - "uplifting animated movies from the 2000s rated above 8"
      - "thrillers directed by Christopher Nolan under 150 mins"

    Returns:
      {
        "genres": List[str],
        "decade": Optional[str],
        "min_year": Optional[int],
        "max_year": Optional[int],
        "max_runtime": Optional[int],
        "min_rating": Optional[float],
        "actors": List[str],
        "directors": List[str],
        "mood": Optional[str],
        "raw_query": str,
        "explanation": List[str]
      }
    """
    q = query.strip()
    result: Dict[str, Any] = {
        "genres": [],
        "decade": None,
        "min_year": None,
        "max_year": None,
        "max_runtime": None,
        "min_rating": None,
        "actors": [],
        "directors": [],
        "mood": None,
        "raw_query": q,
        "explanation": [],
    }

    if not q:
        return result

    q_lower = q.lower()

    # 1. Parse Mood
    mood_patterns = {
        "dark": ["dark", "grim", "gritty", "somber", "gloomy", "bleak"],
        "uplifting": ["uplifting", "inspirational", "inspiring", "heartwarming", "feel-good", "feelgood"],
        "intense": ["intense", "extreme", "edge of your seat", "gripping", "adrenaline"],
        "light-hearted": ["light-hearted", "lighthearted", "fun", "breezy", "hilarious", "cheerful", "cozy"],
        "neutral": ["neutral", "objective", "balanced", "documentary-like"],
    }
    for mood_name, keywords in mood_patterns.items():
        if any(re.search(r"\b" + re.escape(kw) + r"\b", q_lower) for kw in keywords):
            result["mood"] = mood_name.capitalize()
            result["explanation"].append(f"Mood detected: {result['mood']}")
            break

    # 2. Parse Genres & Synonyms
    genre_targets = {
        "Action": ["action", "actions"],
        "Adventure": ["adventure", "adventures"],
        "Animation": ["animation", "animated", "cartoon", "cartoons", "anime"],
        "Comedy": ["comedy", "comedies", "funny", "humorous"],
        "Crime": ["crime", "crimes", "gangster", "gangsters", "mob", "mafia"],
        "Documentary": ["documentary", "documentaries", "docs", "doc"],
        "Drama": ["drama", "dramas", "dramatic"],
        "Family": ["family", "kids", "children"],
        "Fantasy": ["fantasy", "fantasies", "magical"],
        "History": ["history", "historical", "period piece"],
        "Horror": ["horror", "horrors", "scary", "spooky"],
        "Music": ["music", "musical", "musicals"],
        "Mystery": ["mystery", "mysteries", "whodunit", "whodunits"],
        "Romance": ["romance", "romances", "romantic", "romcom", "romcoms", "love story"],
        "Science Fiction": ["sci-fi", "scifi", "science fiction", "science-fiction", "space opera", "cyberpunk"],
        "Thriller": ["thriller", "thrillers", "suspense", "suspenseful", "psychological thriller"],
        "War": ["war", "wars", "military"],
        "Western": ["western", "westerns", "cowboy", "cowboys"],
    }
    found_genres: Set[str] = set()
    for genre_name, triggers in genre_targets.items():
        for trigger in triggers:
            if re.search(r"\b" + re.escape(trigger) + r"\b", q_lower):
                found_genres.add(genre_name)
                break
    if found_genres:
        result["genres"] = sorted(list(found_genres))
        result["explanation"].append(f"Genres: {', '.join(result['genres'])}")

    # 3. Parse Decade (e.g., "90s", "1990s", "eighties", "2000s")
    decade_match = re.search(r"\b(19\d0|20\d0)s\b", q_lower)
    if decade_match:
        dec_str = decade_match.group(1) + "s"
        result["decade"] = dec_str
        dec_num = int(decade_match.group(1))
        result["min_year"] = dec_num
        result["max_year"] = dec_num + 9
        result["explanation"].append(f"Decade: {dec_str} ({dec_num}–{dec_num + 9})")
    else:
        # Check shorthand like "80s", "90s", "70s", "60s", "50s"
        shorthand = re.search(r"\b([5-9]0)s\b", q_lower)
        if shorthand:
            dec_num = 1900 + int(shorthand.group(1))
            result["decade"] = f"{dec_num}s"
            result["min_year"] = dec_num
            result["max_year"] = dec_num + 9
            result["explanation"].append(f"Decade: {dec_num}s ({dec_num}–{dec_num + 9})")
        elif "eighties" in q_lower:
            result["decade"] = "1980s"
            result["min_year"] = 1980
            result["max_year"] = 1989
            result["explanation"].append("Decade: 1980s")
        elif "nineties" in q_lower:
            result["decade"] = "1990s"
            result["min_year"] = 1990
            result["max_year"] = 1999
            result["explanation"].append("Decade: 1990s")
        elif "seventies" in q_lower:
            result["decade"] = "1970s"
            result["min_year"] = 1970
            result["max_year"] = 1979
            result["explanation"].append("Decade: 1970s")

    # 4. Parse Specific Year if no decade found
    if not result["decade"]:
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", q_lower)
        if year_match:
            exact_year = int(year_match.group(1))
            result["min_year"] = exact_year
            result["max_year"] = exact_year
            result["explanation"].append(f"Release Year: {exact_year}")

    # 5. Parse Runtime (e.g., "under 2 hours", "less than 90 minutes", "under 120 mins")
    rt_hours_match = re.search(r"(?:under|less than|within|shorter than)\s+(\d+(?:\.\d+)?)\s*(?:hours|hrs|hr|h)\b", q_lower)
    if rt_hours_match:
        hrs = float(rt_hours_match.group(1))
        max_mins = int(hrs * 60)
        result["max_runtime"] = max_mins
        result["explanation"].append(f"Max Runtime: {max_mins} minutes ({hrs} hrs)")
    else:
        rt_mins_match = re.search(r"(?:under|less than|within|shorter than)\s+(\d+)\s*(?:minutes|mins|min|m)\b", q_lower)
        if rt_mins_match:
            max_mins = int(rt_mins_match.group(1))
            result["max_runtime"] = max_mins
            result["explanation"].append(f"Max Runtime: {max_mins} minutes")

    # 6. Parse Minimum Rating (e.g., "rated above 8", "rating > 7.5", "at least 8 stars", "over 7/10")
    rating_match = re.search(r"(?:rated (?:above|over)|rating (?:above|over|>|>=)|at least|over)\s*(\d+(?:\.\d+)?)\s*(?:stars|/10|/ 10)?", q_lower)
    if rating_match:
        r_val = float(rating_match.group(1))
        if 0.0 <= r_val <= 10.0:
            result["min_rating"] = r_val
            result["explanation"].append(f"Minimum Rating: {r_val}/10")

    # 7. Parse Director (e.g., "directed by Christopher Nolan", "by Quentin Tarantino")
    dir_match = re.search(r"(?:directed by|dir\.? by|by director|by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", q)
    if dir_match:
        director_name = dir_match.group(1).strip()
        result["directors"].append(director_name)
        result["explanation"].append(f"Director: {director_name}")

    # 8. Parse Actor / Cast (e.g., "with Keanu Reeves", "starring Leonardo DiCaprio")
    cast_match = re.search(r"(?:with|starring|featuring|actor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", q)
    if cast_match:
        actor_name = cast_match.group(1).strip()
        # Avoid duplicating if caught by director
        if actor_name not in result["directors"]:
            result["actors"].append(actor_name)
            result["explanation"].append(f"Actor: {actor_name}")

    return result
