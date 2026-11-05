# 🎬 CineMatch — Intelligent Movie Recommendation & Discovery Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-red.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **CineMatch** is an end-to-end, production-grade movie recommendation web application built with **Streamlit**, **scikit-learn**, **NLTK VADER**, **Plotly**, and the **TMDB REST API**. It features content-based filtering, narrative sentiment modeling, a user-tunable hybrid engine, multi-dimensional search filtering with zero-result relaxation diagnostics, a rule-based natural language query parser, centroid-based watchlist taste profiling, and interactive data visualizations.

---

## 🌟 Key Features

1. **Triple Recommendation Engines**:
   - **Content-Based (TF-IDF)**: Weighted term extraction over metadata "soup" (Auteur Director $3\times$, Lead Actors $2\times$, Genres $2\times$, Plot keywords, and overview) with cosine similarity.
   - **Sentiment-Based (NLTK VADER)**: Valence-aware emotional polarity modeling on plot overviews mapping films to mood archetypes (*Uplifting*, *Dark*, *Intense*, *Light-hearted*, *Neutral*).
   - **Hybrid Blending**: User-adjustable slider (default: 70% Content / 30% Mood) with transparent weight reporting.
2. **Explainable AI (XAI)**:
   - Sister-title discovery displaying shared genres, shared keywords, auteur continuity, sentiment closeness, and top driving TF-IDF terms.
3. **Multi-Parameter Discovery Suite**:
   - **Temporal**: Year range slider, decade selector, and relative periods (*"Last 5 years"*, *"This year"*).
   - **Quality**: Minimum star rating (0–10★) and vote count threshold sliders.
   - **Content**: Non-contiguous runtime brackets (`<90`, `90-120`, `120-150`, `150+`), US certification (`G`, `PG`, `PG-13`, `R`, `NC-17`), and original language.
   - **Personnel**: Actor and director autocompletion via `/search/person`.
   - **Logical Genres**: Multi-select with `AND` (all must match), `OR` (any match), and `NOT` (exclusion).
   - **Zero-Result Diagnostic Engine**: Automatically identifies the most restrictive constraint when 0 results match and provides concrete relaxation tips.
4. **Natural Language Query Parser (No Paid LLM)**:
   - Rule-based regex and entity extractor converting prompts like *"dark sci-fi from the 90s under 2 hours with Keanu Reeves"* into structured criteria with an interactive editor.
5. **Watchlist & Taste Centroid Profiling**:
   - Session-state library tracking.
   - Computes composite taste centroid, top genres affinity, preferred decades, and favorite directors.
   - Generates recommendations directly from your aggregate profile vector.
   - CSV export and import for persistent cross-session portability.
6. **Side-by-Side Comparison & Spider Radar Chart**:
   - Select 2–3 movies to benchmark across Rating, Popularity, Runtime, Revenue, and Sentiment on a 5-axis Plotly radar chart.
7. **Cohort Analytics & Visualizations**:
   - Interactive Plotly figures: Ratings by decade, genre trends over time, rating vs. votes scatter, budget vs. box office ROI, and narrative mood distribution.

---

## 🏗️ Project Architecture & File Structure

```text
CineMatch/
├── .streamlit/
│   ├── config.toml             # Cinematic dark theme (near-black #0E1117, crimson accent #E50914)
│   └── secrets.toml.example    # Template for Streamlit secrets
├── tests/
│   ├── __init__.py
│   ├── test_filters.py         # Unit tests for filter validation & TMDB parameter mapping
│   ├── test_nlp.py             # Unit tests for rule-based query parser & VADER sentiment
│   ├── test_recommender.py     # Unit tests for TF-IDF, VADER mood matching & hybrid blending
│   ├── test_watchlist.py       # Unit tests for taste profiling & CSV import/export
│   └── test_combinations.py    # Deep verification of complex multi-filter combinations
├── tmdb_client.py              # TMDB client with Session, backoff, TTL caching & mock fallback
├── features.py                 # Feature engineering: metadata cleaning & weighted text soup
├── nlp.py                      # TF-IDF vectorizer, VADER sentiment, and NL query parser
├── recommender.py              # Content, Sentiment, and Hybrid recommender engines
├── filters.py                  # Structured Filter model, discover builder & zero-result diagnostics
├── watchlist.py                # Session-state watchlist, taste profile vectorizer & CSV handlers
├── viz.py                      # Plotly charts (radar, box plot, scatter, ROI, mood distribution)
├── app.py                      # Streamlit application entry point, sidebar router & movie cards
├── requirements.txt            # Pinned production dependencies
├── .env.example                # Local environment variable template
├── .gitignore                  # Excludes .env, secrets.toml, and cache files
└── README.md                   # Comprehensive technical documentation
```

---

## 🔬 Key Engineering & Algorithmic Decisions

### 1. Why TF-IDF and Cosine Similarity?
TF-IDF downweights ubiquitous stopwords while elevating domain-specific terminology (e.g., `cyberpunk`, `christophernolan`, `heist`). Cosine similarity computes the cosine of the angle between two non-zero vectors in multi-dimensional space, ensuring that longer plot synopses do not artificially outscore concise, punchy loglines.

### 2. Why NLTK VADER for Plot Sentiment?
Plot summaries are concise, emotive, and heavily reliant on context modifiers (*"not terrifying"*, *"extremely grim"*, contrastive *"but"*). VADER handles valence awareness out-of-the-box and provides a normalized `compound` metric from $-1.0$ (extreme negative/dark) to $+1.0$ (extreme positive/uplifting), enabling accurate mood matching without heavy transformer dependencies.

### 3. Server-Side vs. Client-Side Filtering
- **Server-Side (TMDB `/discover/movie`)**:
  - Offloads primary candidate retrieval to TMDB: temporal date boundaries (`primary_release_date.gte/lte`), minimum rating (`vote_average.gte`), minimum votes (`vote_count.gte`), primary genre IDs with `AND` (comma) or `OR` (pipe), and genre exclusions (`without_genres`).
- **Client-Side (Post-Fetch Verification)**:
  - Enforces complex non-contiguous runtime brackets (e.g., selecting `<90 min` AND `150+ min` simultaneously).
  - Verifies verified US certifications from appended `release_dates`.
  - Disambiguates personnel roles (guaranteeing Director vs. Actor distinction).
  - Guarantees 100% adherence to active constraint combinations.

### 4. Parallel Detail Fetching
Basic discovery summaries lack full director, US certification, and keyword data. CineMatch uses Python's `concurrent.futures.ThreadPoolExecutor` (8 workers) to fetch detailed movie records in parallel, reducing candidate pool load times from over 10 seconds to under 1 second.

### 5. Multi-Level Caching & Resilience
- Every API query is cached with `@st.cache_data(ttl=3600)` to respect TMDB rate limits.
- Connection pooling is managed via `requests.Session()` with `urllib3` exponential backoff retrying on HTTP 429 and 5xx errors.
- Includes a high-fidelity offline sample catalog so the application never breaks even without an API key or when network access is constrained.

---

## 🚀 Local Installation & Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12+
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/CineMatch.git
cd CineMatch
```

### 3. Create & Activate Virtual Environment
```bash
# On macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
python -m venv venv
.\venv\Scripts\activate.bat
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure TMDB API Key
1. Obtain a free TMDB v3 API key at [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api).
2. Create your local `.env` file from the example:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` and add your key:
   ```env
   TMDB_API_KEY=your_actual_tmdb_api_key_here
   ```
*(Alternatively, enter your key interactively in the CineMatch sidebar or test with the built-in demo catalog!)*

### 6. Run the Application Locally
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running the Automated Test Suite

CineMatch includes 24 comprehensive unit tests covering filter validation, TMDB query translation, multi-condition combination tests, rule-based natural language parsing, VADER sentiment scoring, TF-IDF cosine similarity, and watchlist serialization.

Run with `pytest`:
```bash
pytest -v
```

Or run using Python's built-in `unittest` runner:
```bash
python -m unittest discover -s tests -v
```

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

Deploying CineMatch to **Streamlit Community Cloud** is fast and free:

1. **Push to GitHub**:
   Ensure your code is committed to a GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Initial CineMatch release"
   git remote add origin https://github.com/your-username/cinematch.git
   git push -u origin main
   ```
   *(Ensure `.env` and `.streamlit/secrets.toml` are in `.gitignore` so your API key is never committed).*

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
   - Click **"New app"**.
   - Select your repository (`cinematch`), branch (`main`), and set **Main file path** to `app.py`.

3. **Configure Secrets**:
   - In the Streamlit Cloud deployment dashboard, click **Advanced Settings** (or go to App Settings > Secrets).
   - Enter your TMDB API Key in TOML format:
     ```toml
     TMDB_API_KEY = "your_actual_tmdb_api_key_here"
     ```
   - Click **Save**.

4. **Automatic Resource Download**:
   - CineMatch automatically downloads the NLTK `vader_lexicon` into the environment on initial boot.
   - Your application will now build and launch at `https://<your-subdomain>.streamlit.app`!

---

## ⚖️ Attribution & Compliance

*This product uses the TMDB API but is not endorsed or certified by TMDB.*
All movie posters, loglines, and metadata are provided courtesy of [The Movie Database (TMDB)](https://www.themoviedb.org/).
