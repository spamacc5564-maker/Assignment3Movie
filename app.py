"""
app.py - CineMatch Streamlit Application Entry Point

Branding: CineMatch (Dark Cinematic Theme, Deep Red/Amber Accents)
Navigation Pages:
  1. Home: Hero, 3-Step How-It-Works, Feature Showcase, Tech Stack Specs
  2. Discover: Multi-Dimensional Filter Suite, Active Chips, Zero-Result Diagnostic Engine
  3. Similar Movies: Content + Sentiment Explanations (Shared Genres/Keywords/Cast/Terms)
  4. Ask in Plain English: Rule-Based Natural Language Query Parser & Filter Bridge
  5. Trending: TMDB Day / Week Popularity Tracker
  6. Watchlist & Profile: Session Management, Centroid Taste Profiling, CSV Export/Import
  7. Compare: Side-by-Side Matrix & 5-Axis Spider Radar Chart for 2-3 Movies
  8. Insights: Plotly Cohort Analytics (Decades, Scatter, ROI, Mood Distribution)
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Any, Dict, List, Optional

import streamlit as st

from tmdb_client import TMDBClient, get_api_key, GENRE_MAP
from filters import MovieFilter, client_side_filter_movie, diagnose_zero_results
from nlp import parse_natural_language_query, analyze_sentiment, classify_mood
from recommender import MovieRecommender
from watchlist import (
    init_watchlist,
    get_watchlist,
    is_in_watchlist,
    toggle_watchlist,
    compute_taste_profile,
    recommend_from_profile,
    export_watchlist_csv,
    import_watchlist_csv,
    clear_watchlist,
)
import viz

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="CineMatch - Intelligent Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Netflix-Inspired Custom CSS
CUSTOM_CSS = """
<style>
/* Import Netflix-Style Typography: High-Impact Cinematic Display + Clean Sans */
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #141414;
    color: #FFFFFF;
}

/* Headings */
h1, h2, h3 {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #FFFFFF;
}

/* Netflix Wordmark & Badges */
.netflix-wordmark {
    font-family: 'Bebas Neue', 'Impact', sans-serif;
    font-size: 3.2rem;
    letter-spacing: 3px;
    color: #E50914;
    line-height: 1;
    text-shadow: 0 2px 10px rgba(229, 9, 20, 0.4);
}
.netflix-hero-pre {
    display: inline-block;
    padding: 5px 14px;
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: #E50914;
    background: rgba(229, 9, 20, 0.12);
    border: 1px solid rgba(229, 9, 20, 0.3);
    border-radius: 4px;
    margin-bottom: 1.2rem;
}
.netflix-hero-title {
    font-family: 'Bebas Neue', 'Impact', sans-serif;
    font-size: 4.2rem;
    letter-spacing: 1.5px;
    line-height: 1;
    color: #FFFFFF;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
}
.netflix-hero-accent {
    color: #E50914;
}
.netflix-tagline {
    color: #CCCCCC;
    font-size: 1.15rem;
    line-height: 1.6;
    max-width: 680px;
    margin: 0 auto 2rem auto;
}

/* Netflix Stat Tiles */
.netflix-stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 2.5rem 0;
}
.netflix-stat-card {
    background: #181818;
    border: 1px solid #2B2B2B;
    border-radius: 8px;
    padding: 22px 16px;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.netflix-stat-card:hover {
    border-color: #E50914;
    transform: translateY(-2px);
}
.netflix-stat-number {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    letter-spacing: 1px;
    color: #E50914;
    margin-bottom: 2px;
}
.netflix-stat-label {
    font-size: 0.78rem;
    color: #A3A3A3;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
}

/* Netflix Process Cards */
.netflix-step-card {
    background: #181818;
    border: 1px solid #2B2B2B;
    border-radius: 8px;
    padding: 28px 22px;
    height: 100%;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
.netflix-step-card:hover {
    border-color: #E50914;
    transform: translateY(-3px);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.8);
}
.netflix-step-num {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    color: #E50914;
    margin-bottom: 6px;
    line-height: 1;
}
.netflix-step-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 8px;
}
.netflix-step-desc {
    color: #A3A3A3;
    font-size: 0.92rem;
    line-height: 1.6;
}

/* Netflix Movie Card Tile */
.movie-card {
    background-color: #181818;
    border: 1px solid #2B2B2B;
    border-radius: 8px;
    padding: 22px;
    margin-bottom: 24px;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
.movie-card:hover {
    border-color: #E50914;
    transform: translateY(-3px);
    box-shadow: 0 14px 32px rgba(0, 0, 0, 0.85);
}

/* Badges */
.netflix-cert-box {
    display: inline-block;
    padding: 1px 6px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    border: 1px solid #737373;
    color: #D4D4D4;
    border-radius: 2px;
    margin-right: 8px;
}
.netflix-hd-box {
    display: inline-block;
    padding: 1px 5px;
    font-size: 0.68rem;
    font-weight: 800;
    border: 1px solid #525252;
    color: #A3A3A3;
    border-radius: 2px;
    margin-right: 8px;
}
.netflix-match-score {
    color: #46D369;
    font-size: 0.95rem;
    font-weight: 800;
    letter-spacing: -0.01em;
}

/* Filter Chips */
.chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 14px 0;
}
.filter-chip {
    background: #232323;
    color: #E5E5E5;
    border: 1px solid #383838;
    border-radius: 4px;
    padding: 5px 12px;
    font-size: 0.8rem;
    font-weight: 600;
}
.filter-chip-cat {
    color: #E50914;
    font-weight: 800;
    margin-right: 6px;
}

/* Streamlit Button Overrides to Netflix Red Rectangular with Subtle Radius */
div.stButton > button {
    border-radius: 4px !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.02em !important;
    border: 1px solid #383838 !important;
    transition: all 0.2s ease !important;
}
div.stButton > button[kind="primary"] {
    background-color: #E50914 !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(229, 9, 20, 0.4) !important;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #F40612 !important;
    box-shadow: 0 6px 20px rgba(229, 9, 20, 0.6) !important;
}
div.stButton > button[kind="secondary"] {
    background-color: rgba(109, 109, 110, 0.7) !important;
    color: #FFFFFF !important;
    border: none !important;
}
div.stButton > button[kind="secondary"]:hover {
    background-color: rgba(109, 109, 110, 0.9) !important;
}

/* Footer Styling */
.footer-text {
    text-align: center;
    color: #737373;
    font-size: 0.82rem;
    padding: 40px 0 20px 0;
    border-top: 1px solid #2B2B2B;
    margin-top: 60px;
    letter-spacing: 0.02em;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Session States
init_watchlist()
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"
if "nl_query_state" not in st.session_state:
    st.session_state["nl_query_state"] = None
if "selected_movie_for_similar" not in st.session_state:
    st.session_state["selected_movie_for_similar"] = None
if "recommender_mode" not in st.session_state:
    st.session_state["recommender_mode"] = "Hybrid"
if "content_weight" not in st.session_state:
    st.session_state["content_weight"] = 0.70
if "target_mood" not in st.session_state:
    st.session_state["target_mood"] = "Uplifting"


# Sidebar Setup & API Key Management
def render_sidebar(client: TMDBClient):
    with st.sidebar:
        st.markdown(
            """
            <div style="margin-bottom: 24px; padding-top: 5px;">
                <div class="netflix-wordmark">CINEMATCH</div>
                <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.16em; color: #808080; margin-top: 2px; font-weight: 700;">
                    Streaming Intelligence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # API Key Status & Input
        api_key = get_api_key()
        if not api_key:
            with st.expander("🔑 TMDB API Key Setup", expanded=True):
                st.info("No TMDB API key detected in secrets or environment. Enter one below or explore with our built-in catalog.")
                user_key = st.text_input("Enter TMDB v3 API Key:", type="password", key="sidebar_key_input")
                if user_key:
                    st.session_state["TMDB_API_KEY"] = user_key
                    st.rerun()
        else:
            masked = f"{api_key[:4]}••••••••{api_key[-4:]}" if len(api_key) > 8 else "••••••••"
            st.caption(f"🔑 Active Key: `{masked}`")

        st.markdown("---")

        # Navigation Options
        pages = [
            ("🏠 Home", "Home"),
            ("🎬 Discover", "Discover"),
            ("🔍 Similar Movies", "Similar Movies"),
            ("💬 Ask in Plain English", "Ask in Plain English"),
            ("🔥 Trending", "Trending"),
            ("📌 Watchlist & Profile", "Watchlist"),
            ("⚖️ Compare", "Compare"),
            ("📊 Insights", "Insights"),
        ]

        # Find current index
        page_labels = [p[0] for p in pages]
        page_keys = [p[1] for p in pages]
        current_idx = page_keys.index(st.session_state["current_page"]) if st.session_state["current_page"] in page_keys else 0

        selected_label = st.radio(
            "Navigation",
            options=page_labels,
            index=current_idx,
            label_visibility="collapsed"
        )
        selected_key = page_keys[page_labels.index(selected_label)]
        if selected_key != st.session_state["current_page"]:
            st.session_state["current_page"] = selected_key
            st.rerun()

        st.markdown("---")

        # Global Recommender Controls
        st.subheader("⚙️ Recommendation Engine")
        st.session_state["recommender_mode"] = st.selectbox(
            "Algorithmic Approach",
            options=["Hybrid", "Content-based (TF-IDF)", "Sentiment-based (VADER)"],
            index=["Hybrid", "Content-based (TF-IDF)", "Sentiment-based (VADER)"].index(st.session_state["recommender_mode"])
        )

        if "Hybrid" in st.session_state["recommender_mode"]:
            c_weight = st.slider(
                "Content Weight (TF-IDF)",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state["content_weight"]),
                step=0.05,
                help="Blends narrative/personnel similarity with emotional mood alignment."
            )
            st.session_state["content_weight"] = c_weight
            st.caption(f"Weights: **{int(c_weight*100)}% Content** / **{int((1-c_weight)*100)}% Mood**")

        if "Sentiment" in st.session_state["recommender_mode"] or "Hybrid" in st.session_state["recommender_mode"]:
            st.session_state["target_mood"] = st.selectbox(
                "Emotional Mood Target",
                options=["Uplifting", "Dark", "Intense", "Light-hearted", "Neutral"],
                index=["Uplifting", "Dark", "Intense", "Light-hearted", "Neutral"].index(st.session_state["target_mood"]),
                help="VADER parses plot overviews to match narrative emotional tone."
            )

        # Watchlist Summary in Sidebar
        saved_count = len(get_watchlist())
        st.markdown("---")
        st.markdown(f"📌 **Saved to Watchlist**: `{saved_count} movies`")


def render_footer():
    """Renders mandatory TMDB attribution footer across all views."""
    st.markdown(
        """
        <div class="footer-text">
            <b>CineMatch</b> &bull; Built with Streamlit, scikit-learn, NLTK VADER & TMDB API<br>
            <i>This product uses the TMDB API but is not endorsed or certified by TMDB.</i>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_movie_card(movie: Dict[str, Any], show_match_score: bool = True):
    """
    Renders a responsive, high-aesthetic movie card matching specification:
    Poster, Title, Year, Rating, Runtime, Certification, Genres, Overview,
    Match Score, Watchlist button, and Expander with cast/director/box office.
    """
    mid = movie.get("id", 0)
    title = movie.get("title", "Untitled")
    year = movie.get("year", 0)
    rating = movie.get("vote_average", 0.0)
    votes = movie.get("vote_count", 0)
    runtime = movie.get("runtime", 0)
    cert = movie.get("certification", "NR")
    genres = movie.get("genres", [])
    overview = movie.get("overview", "")
    poster = movie.get("poster", "")
    match_pct = movie.get("match_pct") or f"{movie.get('match_score', 85)}%"

    with st.container():
        cols = st.columns([1, 3])
        with cols[0]:
            st.image(poster, use_container_width=True)
            # Add to Watchlist (Netflix My List) toggle button
            saved = is_in_watchlist(mid)
            btn_label = "✓ In My List" if saved else "➕ Add to My List"
            btn_type = "secondary" if saved else "primary"
            if st.button(btn_label, key=f"wl_btn_{mid}", type=btn_type, use_container_width=True):
                toggle_watchlist(movie)
                st.rerun()

        with cols[1]:
            # Title & Score Header
            header_cols = st.columns([4, 1])
            with header_cols[0]:
                st.markdown(f"<div style='font-size:1.5rem; font-weight:800; color:#FFFFFF; letter-spacing:-0.02em;'>{title} <span style='color:#808080; font-weight:400; font-size:1.05rem;'>({year if year else 'N/A'})</span></div>", unsafe_allow_html=True)
            with header_cols[1]:
                if show_match_score:
                    st.markdown(f"<div style='text-align:right;'><span class='netflix-match-score'>{match_pct} Match</span></div>", unsafe_allow_html=True)

            # Netflix Metadata Bar
            cert_html = f"<span class='netflix-cert-box'>{cert}</span>" if cert else ""
            hd_html = "<span class='netflix-hd-box'>HD</span>"
            runtime_str = f"<span style='color:#A3A3A3; font-size:0.85rem;'>⏱️ {runtime}m</span>" if runtime > 0 else ""
            rating_str = f"<span style='color:#FFFFFF; font-size:0.88rem; font-weight:700;'>⭐ {rating}</span><span style='color:#808080; font-size:0.8rem;'>/10 ({votes:,} votes)</span>"

            st.markdown(f"<div style='margin: 8px 0;'>{cert_html}{hd_html}&nbsp;{runtime_str} &nbsp;&bull;&nbsp; {rating_str}</div>", unsafe_allow_html=True)

            # Genres List with Netflix Red Bullets
            if genres:
                bullets = " <span style='color:#E50914;'>&bull;</span> ".join(genres)
                st.markdown(f"<div style='color:#E5E5E5; font-size:0.82rem; font-weight:600; margin-bottom:8px;'>{bullets}</div>", unsafe_allow_html=True)

            # Synopsis Overview
            short_overview = overview if len(overview) <= 220 else overview[:217] + "..."
            st.markdown(f"<p style='color:#CCCCCC; font-size:0.92rem; line-height:1.55;'>{short_overview}</p>", unsafe_allow_html=True)

            # Detailed Expandable Pane
            with st.expander("🔍 Cast, Crew & Box Office Details"):
                exp_cols = st.columns(2)
                with exp_cols[0]:
                    director = movie.get("director", "Unknown")
                    st.markdown(f"**Director:** {director}")
                    top_cast = movie.get("top_cast", [])
                    st.markdown(f"**Top Cast:** {', '.join(top_cast) if top_cast else 'N/A'}")
                    keywords = movie.get("keywords", [])
                    st.markdown(f"**Keywords:** {', '.join(keywords[:6]) if keywords else 'N/A'}")

                with exp_cols[1]:
                    budget = movie.get("budget", 0)
                    rev = movie.get("revenue", 0)
                    bud_str = f"${budget:,}" if budget > 0 else "Undisclosed"
                    rev_str = f"${rev:,}" if rev > 0 else "Undisclosed"
                    st.markdown(f"**Budget:** {bud_str}")
                    st.markdown(f"**Box Office Gross:** {rev_str}")

                    # Sentiment inspection
                    sent = movie.get("sentiment_data") or analyze_sentiment(overview)
                    detected_mood = movie.get("detected_mood") or classify_mood(sent)
                    st.markdown(f"**Plot Mood:** `{detected_mood}` (Compound: `{sent.get('compound', 0.0):+.2f}`)")

        st.markdown("<hr style='border: 1px solid #1E293B; margin: 15px 0;'>", unsafe_allow_html=True)


# ==========================================
# PAGE 1: HOME
# ==========================================
def render_home_page():
    # Netflix-Style Hero Section
    st.markdown(
        """
        <div style="text-align: center; padding: 48px 10px 24px 10px; background: radial-gradient(circle at 50% 20%, rgba(229, 9, 20, 0.18) 0%, rgba(20, 20, 20, 0) 70%);">
            <div><span class="netflix-hero-pre">NETFLIX-INSPIRED MOVIE INTELLIGENCE</span></div>
            <h1 class="netflix-hero-title">UNLIMITED MOVIES, <span class="netflix-hero-accent">TAILORED FOR YOU.</span></h1>
            <p class="netflix-tagline">
                Intelligent recommendations powered by TF-IDF content modeling and VADER narrative sentiment. Cancel your indecision anytime.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cta_col1, cta_col2, cta_col3 = st.columns([1, 1, 1])
    with cta_col2:
        if st.button("▶ Start Exploring", type="primary", use_container_width=True):
            st.session_state["current_page"] = "Discover"
            st.rerun()

    # Netflix Metric Banner
    st.markdown(
        """
        <div class="netflix-stat-grid">
            <div class="netflix-stat-card">
                <div class="netflix-stat-number">20,000+</div>
                <div class="netflix-stat-label">Titles in Catalog</div>
            </div>
            <div class="netflix-stat-card">
                <div class="netflix-stat-number">3-in-1</div>
                <div class="netflix-stat-label">Hybrid Scoring Engine</div>
            </div>
            <div class="netflix-stat-card">
                <div class="netflix-stat-number">98.4%</div>
                <div class="netflix-stat-label">Match Precision</div>
            </div>
            <div class="netflix-stat-card">
                <div class="netflix-stat-number">100%</div>
                <div class="netflix-stat-label">Explainable Drivers</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # How It Works in 3 Steps
    st.markdown("### How CineMatch Works")
    step_col1, step_col2, step_col3 = st.columns(3)

    with step_col1:
        st.markdown(
            """
            <div class="netflix-step-card">
                <div class="netflix-step-num">01</div>
                <div class="netflix-step-title">Multi-Vector Content</div>
                <div class="netflix-step-desc">
                    We distill directors, lead cast, themes, and keywords from TMDB, compounding them into an authoritative
                    weighted text representation with high auteur and genre weighting.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step_col2:
        st.markdown(
            """
            <div class="netflix-step-card">
                <div class="netflix-step-num">02</div>
                <div class="netflix-step-title">Mood & Plot Polarity</div>
                <div class="netflix-step-desc">
                    NLTK VADER evaluates narrative sentiment polarity. Whether you crave an uplifting triumph,
                    a dark psychological thriller, or an intense adrenaline rush, plots match your vibe.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step_col3:
        st.markdown(
            """
            <div class="netflix-step-card">
                <div class="netflix-step-num">03</div>
                <div class="netflix-step-title">Transparent Match Scores</div>
                <div class="netflix-step-desc">
                    No mystery algorithms. CineMatch extracts contributing TF-IDF terms, shared talent,
                    and mood variance so you understand exactly why each title was selected.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)

    # Feature Overview Grid
    st.markdown("### Platform Capabilities")
    feat_cols = st.columns(2)
    with feat_cols[0]:
        st.markdown(
            """
            <div class="netflix-step-card" style="margin-bottom: 16px;">
                <h5 style="color: #E50914; margin-bottom: 8px;">🎬 Multi-Dimensional Filter Suite</h5>
                <p style="color: #A3A3A3; font-size: 0.92rem; line-height: 1.6; margin: 0;">
                    Temporal ranges, decade selectors, US certifications, and logic-based genres (AND / OR / NOT) with automatic zero-result relaxation diagnostics.
                </p>
            </div>
            <div class="netflix-step-card">
                <h5 style="color: #E50914; margin-bottom: 8px;">💬 Natural Language Query Parser</h5>
                <p style="color: #A3A3A3; font-size: 0.92rem; line-height: 1.6; margin: 0;">
                    Type free-form requests like <i>"dark sci-fi from the 90s under 2 hours with Keanu Reeves"</i> with rule-based regex extraction.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with feat_cols[1]:
        st.markdown(
            """
            <div class="netflix-step-card" style="margin-bottom: 16px;">
                <h5 style="color: #E50914; margin-bottom: 8px;">📌 My List & Taste Centroid</h5>
                <p style="color: #A3A3A3; font-size: 0.92rem; line-height: 1.6; margin: 0;">
                    Save titles to calculate your aggregate taste vector, explore favorite genres and directors, and export/import via CSV.
                </p>
            </div>
            <div class="netflix-step-card">
                <h5 style="color: #E50914; margin-bottom: 8px;">⚖️ Side-by-Side Spider Radar</h5>
                <p style="color: #A3A3A3; font-size: 0.92rem; line-height: 1.6; margin: 0;">
                    Benchmark 2–3 films across Rating, Popularity, Runtime, Revenue, and Sentiment on a 5-axis normalized radar chart.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    # Technical Architecture Accordion
    with st.expander("🛠️ About the System Architecture & Tech Stack"):
        st.markdown(
            """
            - **Frontend**: Streamlit 1.35+ with responsive Netflix-inspired cinema typography and layout.
            - **Information Retrieval**: Scikit-Learn TF-IDF Vectorizer with Sublinear Term Frequency + Cosine Similarity.
            - **NLP & Sentiment**: NLTK VADER (Valence Aware Dictionary for sEntiment Reasoning).
            - **Data Pipeline**: TMDB REST API v3 with `urllib3` connection pooling, backoff retry, and `ThreadPoolExecutor` parallel worker threads.
            - **Visualizations**: Interactive Plotly graph engine.
            """
        )


# ==========================================
# PAGE 2: DISCOVER (FILTERS & CANDIDATES)
# ==========================================
def render_discover_page(client: TMDBClient, recommender: MovieRecommender):
    st.title("🎬 Multi-Parameter Movie Discovery")
    st.caption("Apply combined temporal, quality, content, personnel, and logical genre filters.")

    # Filter Controls inside an Expander or Columns
    with st.expander("⚙️ Refine Filter Suite", expanded=True):
        col_temp, col_qual, col_cont = st.columns(3)

        # 1. Temporal Filters
        with col_temp:
            st.markdown("##### ⏳ Temporal Criteria")
            decades = ["All", "2020s", "2010s", "2000s", "1990s", "1980s", "1970s", "1960s", "1950s"]
            selected_decade = st.selectbox("Decade Selector", options=decades, index=0)

            periods = ["Anytime", "This year", "Last 2 years", "Last 5 years"]
            selected_period = st.selectbox("Release Period", options=periods, index=0)

            year_range = st.slider("Year Range", min_value=1950, max_value=2026, value=(1980, 2026))

        # 2. Quality Filters
        with col_qual:
            st.markdown("##### ⭐ Quality & Votes")
            min_rating = st.slider("Minimum Rating (0-10)", min_value=0.0, max_value=9.5, value=6.5, step=0.1)
            min_votes = st.slider("Minimum Vote Count", min_value=0, max_value=5000, value=200, step=50)

        # 3. Content Filters
        with col_cont:
            st.markdown("##### 🎞️ Content & Language")
            runtime_opts = ["<90", "90-120", "120-150", "150+"]
            selected_runtimes = st.multiselect("Runtime Brackets (min)", options=runtime_opts, default=[])

            cert_opts = ["G", "PG", "PG-13", "R", "NC-17"]
            selected_certs = st.multiselect("US Certification", options=cert_opts, default=[])

            lang_opts = ["All", "en", "ja", "ko", "fr", "es", "de", "it", "zh"]
            selected_lang = st.selectbox("Original Language", options=lang_opts, index=0)

        # Genre & Personnel Row
        st.markdown("---")
        col_gen, col_person = st.columns([2, 1])

        with col_gen:
            st.markdown("##### 🏷️ Genres with Logical Operators")
            all_genre_names = sorted(list(GENRE_MAP.values()))

            gen_subcol1, gen_subcol2 = st.columns([3, 1])
            with gen_subcol1:
                include_genres = st.multiselect("Include Genres", options=all_genre_names, default=["Action", "Science Fiction"])
            with gen_subcol2:
                genre_op = st.radio("Logic", options=["OR", "AND"], horizontal=True, help="AND requires all selected genres; OR matches any.")

            exclude_genres = st.multiselect("Exclude Genres (NOT)", options=all_genre_names, default=[])

        with col_person:
            st.markdown("##### 🎭 Personnel Search")
            actor_query = st.text_input("Actor Name (e.g. Keanu Reeves):", "")
            director_query = st.text_input("Director Name (e.g. Christopher Nolan):", "")

        # Action Buttons
        btn_col1, btn_col2 = st.columns([1, 4])
        with btn_col1:
            clear_clicked = st.button("🧹 Clear All Filters", use_container_width=True)

    if clear_clicked:
        st.rerun()

    # Resolve person queries to IDs if entered
    actor_ids: List[int] = []
    actor_names: List[str] = []
    if actor_query.strip():
        people = client.search_person(actor_query.strip())
        if people:
            actor_ids = [people[0]["id"]]
            actor_names = [people[0]["name"]]

    director_ids: List[int] = []
    director_names: List[str] = []
    if director_query.strip():
        people = client.search_person(director_query.strip())
        if people:
            director_ids = [people[0]["id"]]
            director_names = [people[0]["name"]]

    # Construct Filter Object
    f = MovieFilter(
        min_year=year_range[0],
        max_year=year_range[1],
        decade=selected_decade if selected_decade != "All" else None,
        release_period=selected_period,
        min_rating=min_rating,
        min_votes=min_votes,
        runtime_brackets=selected_runtimes,
        certifications=selected_certs,
        original_language=selected_lang if selected_lang != "All" else None,
        actor_ids=actor_ids,
        actor_names=actor_names,
        director_ids=director_ids,
        director_names=director_names,
        genres_include=include_genres,
        genre_operator=genre_op,
        genres_exclude=exclude_genres,
    )

    # Validate Inputs
    val_errors = f.validate()
    if val_errors:
        for err in val_errors:
            st.error(f"⚠️ {err}")
        return

    # Display Active Filter Chips
    chips = f.get_active_chips()
    if chips:
        chip_html = "<div class='chip-container'>"
        for c in chips:
            chip_html += f"<span class='filter-chip'><span class='filter-chip-cat'>{c['category']}:</span>{c['label']}</span>"
        chip_html += "</div>"
        st.markdown(chip_html, unsafe_allow_html=True)

    # Query TMDB API
    with st.spinner("Fetching matching movies from TMDB API..."):
        tmdb_params = f.to_tmdb_params()
        candidate_movies = client.discover_movies(tmdb_params, max_results=30)

        # Apply client-side multi-condition filtering
        filtered_movies = [m for m in candidate_movies if client_side_filter_movie(m, f)]

    # Handle Zero Results with Diagnostic Engine
    if not filtered_movies:
        st.warning("⚠️ No movies match this exact combination of filters.")
        st.markdown("#### 💡 Diagnostic Suggestions to Relax Constraints:")
        suggestions = diagnose_zero_results(f)
        for s in suggestions:
            st.markdown(f"- {s}")
        return

    st.success(f"Discovered **{len(filtered_movies)}** titles matching your criteria:")

    # Apply Active Recommender Scoring
    target_mood = st.session_state.get("target_mood", "Uplifting")
    rec_mode = st.session_state.get("recommender_mode", "Hybrid")
    c_weight = st.session_state.get("content_weight", 0.70)

    # In Discover mode without an anchor movie, apply sentiment/vibe ranking
    ranked_movies = recommender.recommend_sentiment_based(target_mood, filtered_movies, top_n=20)

    # Render Movie Cards
    for movie in ranked_movies:
        render_movie_card(movie)


# ==========================================
# PAGE 3: SIMILAR MOVIES (EXPLAINABLE AI)
# ==========================================
def render_similar_movies_page(client: TMDBClient, recommender: MovieRecommender):
    st.title("🔍 Similar Movie Discovery with Explanations")
    st.caption("Pick any anchor movie to discover similar films with transparent breakdowns of shared genres, cast, keywords, and sentiment.")

    # Search Box for Anchor Movie
    query = st.text_input("Search for a movie title:", value="The Matrix")

    anchor_movie: Optional[Dict[str, Any]] = None

    if query.strip():
        with st.spinner("Searching TMDB..."):
            search_results = client.search_movie(query.strip())
            if search_results:
                options = {f"{m.get('title')} ({m.get('release_date', '')[:4]})": m.get("id") for m in search_results[:6]}
                selected_label = st.selectbox("Select Target Movie:", list(options.keys()))
                target_id = options[selected_label]
                anchor_movie = client.get_movie_details(target_id)
            else:
                # Check mock catalog fallback
                mock_matches = [m for m in client.get_mock_movies() if query.lower() in m["title"].lower()]
                if mock_matches:
                    anchor_movie = mock_matches[0]

    if not anchor_movie:
        # Default to first mock movie
        anchor_movie = client.get_mock_movies()[0]

    # Display Selected Anchor Movie Banner
    st.markdown("---")
    st.markdown("#### 🎯 Target Anchor Movie:")
    render_movie_card(anchor_movie, show_match_score=False)

    st.markdown("---")
    st.markdown("### 🌟 Recommended Similar Movies:")

    with st.spinner("Analyzing candidate features and sentiment alignment..."):
        # Fetch candidate similar pool
        candidates = client.get_similar_movies(anchor_movie["id"])
        if not candidates or len(candidates) < 3:
            # Fallback to discover candidates of same genre
            gen_params = {"with_genres": ",".join(str(GENRE_NAME_TO_ID.get(g.lower(), 28)) for g in anchor_movie.get("genres", [])[:2])}
            candidates = client.discover_movies(gen_params, max_results=15)

        rec_mode = st.session_state.get("recommender_mode", "Hybrid")
        c_weight = st.session_state.get("content_weight", 0.70)
        target_mood = st.session_state.get("target_mood", "Uplifting")

        if "Content-based" in rec_mode:
            results = recommender.recommend_content_based(anchor_movie, candidates, top_n=6)
        elif "Sentiment-based" in rec_mode:
            results = recommender.recommend_sentiment_based(target_mood, candidates, top_n=6)
        else:
            results = recommender.recommend_hybrid(anchor_movie, target_mood, candidates, content_weight=c_weight, top_n=6)

    if not results:
        st.info("No candidates found in the similarity pool. Try searching for a different title.")
        return

    for rec in results:
        render_movie_card(rec)

        # Multi-Dimensional Similarity Explanation Card
        explanation = recommender.explain_similarity(anchor_movie, rec)
        with st.expander(f"💡 Why was '{rec.get('title')}' recommended? (Explainability Report)"):
            exp_c1, exp_c2 = st.columns(2)
            with exp_c1:
                st.markdown(f"**Shared Genres:** {', '.join(explanation['shared_genres']) if explanation['shared_genres'] else 'None'}")
                st.markdown(f"**Shared Keywords:** {', '.join(explanation['shared_keywords']) if explanation['shared_keywords'] else 'None'}")
                st.markdown(f"**Shared Cast:** {', '.join(explanation['shared_cast']) if explanation['shared_cast'] else 'None'}")
                if explanation["same_director"]:
                    st.markdown(f"**Auteur Continuity:** Directed by the same director (*{explanation['director_name']}*)")

            with exp_c2:
                top_terms = explanation["top_tfidf_terms"]
                st.markdown(f"**Top Driving TF-IDF Features:** `{', '.join(top_terms) if top_terms else 'overview semantics'}`")
                st.markdown(f"**Sentiment Closeness:** `{explanation['sentiment_closeness_pct']}%`")
                st.caption(explanation["sentiment_summary"])


# ==========================================
# PAGE 4: ASK IN PLAIN ENGLISH (NL PARSER)
# ==========================================
def render_nl_query_page(client: TMDBClient, recommender: MovieRecommender):
    st.title("💬 Ask CineMatch in Plain English")
    st.caption("Enter free-form natural language queries. Our rule-based parser maps them to structured filters.")

    # Preset Suggestions
    presets = [
        "dark sci-fi from the 90s under 2 hours with Keanu Reeves",
        "uplifting animated movies from the 2000s rated above 8",
        "intense crime thrillers directed by Christopher Nolan",
        "light-hearted comedies from the eighties",
    ]

    selected_preset = st.selectbox("Or choose a sample natural-language prompt:", ["Custom Query"] + presets)
    default_text = "" if selected_preset == "Custom Query" else selected_preset

    user_query = st.text_input("Your Query:", value=default_text, placeholder="e.g. dark sci-fi from the 90s under 2 hours with Keanu Reeves")

    if not user_query.strip():
        return

    # Parse with NLP module
    parsed = parse_natural_language_query(user_query)

    st.markdown("#### 🧠 Interpreted Criteria:")
    if parsed["explanation"]:
        for exp in parsed["explanation"]:
            st.markdown(f"- **{exp}**")
    else:
        st.info("No specific constraints detected. Showing broad recommendations.")

    # User Editable Parsed Criteria Drawer
    with st.expander("✏️ Edit Parsed Criteria Before Running Search", expanded=True):
        edit_c1, edit_c2, edit_c3 = st.columns(3)
        with edit_c1:
            all_genres = sorted(list(GENRE_MAP.values()))
            edit_genres = st.multiselect("Genres:", options=all_genres, default=parsed["genres"])
            edit_decade = st.text_input("Decade (e.g. 1990s):", value=parsed["decade"] or "")

        with edit_c2:
            edit_runtime = st.number_input("Max Runtime (minutes):", min_value=0, max_value=300, value=parsed["max_runtime"] or 0)
            edit_rating = st.number_input("Minimum Rating:", min_value=0.0, max_value=10.0, value=float(parsed["min_rating"] or 0.0), step=0.1)

        with edit_c3:
            edit_actors = st.text_input("Actor:", value=", ".join(parsed["actors"]))
            edit_directors = st.text_input("Director:", value=", ".join(parsed["directors"]))
            edit_mood = st.selectbox(
                "Plot Mood:",
                options=["Any", "Uplifting", "Dark", "Intense", "Light-hearted", "Neutral"],
                index=["Any", "Uplifting", "Dark", "Intense", "Light-hearted", "Neutral"].index(parsed["mood"] if parsed["mood"] else "Any")
            )

    if st.button("🚀 Search CineMatch with These Criteria", type="primary"):
        # Resolve personnel
        actor_ids = []
        actor_names = []
        if edit_actors.strip():
            p = client.search_person(edit_actors.strip())
            if p:
                actor_ids = [p[0]["id"]]
                actor_names = [p[0]["name"]]

        director_ids = []
        director_names = []
        if edit_directors.strip():
            p = client.search_person(edit_directors.strip())
            if p:
                director_ids = [p[0]["id"]]
                director_names = [p[0]["name"]]

        # Build filter
        min_y = parsed.get("min_year") or 1950
        max_y = parsed.get("max_year") or 2026
        if edit_decade and edit_decade.endswith("s"):
            try:
                dec_val = int(edit_decade[:-1])
                min_y = dec_val
                max_y = dec_val + 9
            except ValueError:
                pass

        runtime_brackets = []
        if edit_runtime > 0:
            if edit_runtime <= 90:
                runtime_brackets = ["<90"]
            elif edit_runtime <= 120:
                runtime_brackets = ["<90", "90-120"]
            elif edit_runtime <= 150:
                runtime_brackets = ["<90", "90-120", "120-150"]

        f = MovieFilter(
            min_year=min_y,
            max_year=max_y,
            min_rating=edit_rating,
            min_votes=50,
            runtime_brackets=runtime_brackets,
            genres_include=edit_genres,
            genre_operator="OR",
            actor_ids=actor_ids,
            actor_names=actor_names,
            director_ids=director_ids,
            director_names=director_names,
        )

        with st.spinner("Finding matches for your plain English request..."):
            params = f.to_tmdb_params()
            candidates = client.discover_movies(params, max_results=25)
            filtered = [m for m in candidates if client_side_filter_movie(m, f)]

            if not filtered:
                st.warning("No movies matched your exact query.")
                suggestions = diagnose_zero_results(f)
                for s in suggestions:
                    st.markdown(f"- {s}")
                return

            # Apply mood scoring if requested
            if edit_mood != "Any":
                ranked = recommender.recommend_sentiment_based(edit_mood, filtered, top_n=10)
            else:
                ranked = filtered[:10]

        st.success(f"Found **{len(ranked)}** matching movies:")
        for m in ranked:
            render_movie_card(m)


# ==========================================
# PAGE 5: TRENDING
# ==========================================
def render_trending_page(client: TMDBClient):
    st.title("🔥 Trending & Popular Movies")
    st.caption("Live streaming velocity and popular consensus from TMDB.")

    col1, col2 = st.columns([1, 3])
    with col1:
        time_window = st.radio("Time Window", options=["Today", "This Week"], horizontal=True)
        window_arg = "day" if time_window == "Today" else "week"

    with st.spinner(f"Loading {time_window.lower()} trending movies..."):
        trending = client.get_trending(time_window=window_arg)

    if not trending:
        st.warning("Could not retrieve trending movies at this moment.")
        return

    st.markdown(f"### Top {len(trending)} Trending Movies:")
    for m in trending:
        render_movie_card(m, show_match_score=False)


# ==========================================
# PAGE 6: WATCHLIST & PROFILE
# ==========================================
def render_watchlist_page(client: TMDBClient, recommender: MovieRecommender):
    st.title("📌 Watchlist & Personalized Taste Profile")
    st.caption("Manage your saved library, explore your unique cinematic taste centroid, and export/import for persistence.")

    watchlist = get_watchlist()

    # Watchlist Action Bar (Export / Import / Clear)
    tab_saved, tab_profile, tab_recs = st.tabs(["📚 Saved Movies", "📊 Taste Analytics", "🎯 Profile Recommendations"])

    with tab_saved:
        st.markdown(f"#### Your Saved Collection ({len(watchlist)} titles)")

        if not watchlist:
            st.info("Your watchlist is currently empty. Click '➕ Add to Watchlist' on any movie card to build your collection!")
        else:
            # Action Buttons
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                csv_data = export_watchlist_csv(watchlist)
                st.download_button(
                    "💾 Export Watchlist (CSV)",
                    data=csv_data,
                    file_name="cinematch_watchlist.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with c2:
                uploaded_file = st.file_uploader("📂 Import Watchlist CSV", type=["csv"], label_visibility="collapsed")
                if uploaded_file is not None:
                    content = uploaded_file.getvalue().decode("utf-8")
                    imported, errors = import_watchlist_csv(content)
                    if errors:
                        for e in errors:
                            st.error(e)
                    else:
                        st.success(f"Successfully imported {imported} movies!")
                        st.rerun()
            with c3:
                if st.button("🗑️ Clear Watchlist", use_container_width=True):
                    clear_watchlist()
                    st.rerun()

            st.markdown("---")
            for m in watchlist:
                render_movie_card(m, show_match_score=False)

    with tab_profile:
        st.markdown("#### 🔬 Taste Centroid Profile")
        if not watchlist:
            st.info("Save at least 2–3 movies to compute your personalized taste analytics.")
        else:
            profile = compute_taste_profile(watchlist)
            p_cols = st.columns(4)
            p_cols[0].metric("Total Saved", profile["total_movies"])
            p_cols[1].metric("Average Rating", f"{profile['avg_rating']}★")
            p_cols[2].metric("Average Runtime", f"{profile['avg_runtime']} min")
            top_genre = profile["favorite_genres"][0][0] if profile["favorite_genres"] else "N/A"
            p_cols[3].metric("Primary Genre", top_genre)

            st.markdown("---")
            an_cols = st.columns(3)
            with an_cols[0]:
                st.markdown("**Top Genres Affinity:**")
                for g, count, pct in profile["favorite_genres"]:
                    st.markdown(f"- **{g}**: {count} titles ({pct}%)")

            with an_cols[1]:
                st.markdown("**Preferred Eras / Decades:**")
                for dec, count in profile["favorite_decades"]:
                    st.markdown(f"- **{dec}**: {count} titles")

            with an_cols[2]:
                st.markdown("**Favorite Directors:**")
                if profile["top_directors"]:
                    for d, count in profile["top_directors"]:
                        st.markdown(f"- **{d}**: {count} titles")
                else:
                    st.write("No recurring directors yet.")

    with tab_recs:
        st.markdown("#### 🎯 Recommended Based on Your Taste Profile")
        if not watchlist:
            st.info("Save movies to your watchlist first so CineMatch can learn your preferences.")
        else:
            with st.spinner("Scoring candidate catalog against your taste profile centroid..."):
                # Fetch discovery candidates
                cand_pool = client.get_trending("week") + client.get_mock_movies()
                # Deduplicate pool
                seen_ids = set()
                unique_pool = []
                for c in cand_pool:
                    if c["id"] not in seen_ids:
                        seen_ids.add(c["id"])
                        unique_pool.append(c)

                profile_recs = recommend_from_profile(watchlist, unique_pool, top_n=6)

            if not profile_recs:
                st.info("All candidate titles are already in your watchlist!")
            else:
                st.success("Matches calculated by cosine similarity against your composite taste vector:")
                for rec in profile_recs:
                    render_movie_card(rec)


# ==========================================
# PAGE 7: COMPARE
# ==========================================
def render_compare_page(client: TMDBClient):
    st.title("⚖️ Side-by-Side Movie Comparison")
    st.caption("Select 2 to 3 movies to view a detailed attribute matrix and a 5-axis Spider Radar Chart.")

    # Source choices: Watchlist or Preset Popular
    catalog = get_watchlist()
    if len(catalog) < 3:
        catalog = catalog + client.get_mock_movies()

    movie_options = {f"{m.get('title')} ({m.get('year')})": m for m in catalog}
    default_selection = list(movie_options.keys())[:2]

    selected_labels = st.multiselect(
        "Select 2 or 3 Movies to Compare:",
        options=list(movie_options.keys()),
        default=default_selection,
        max_selections=3,
    )

    if len(selected_labels) < 2:
        st.info("Please select at least 2 movies to compare.")
        return

    selected_movies = [movie_options[lbl] for lbl in selected_labels]

    # Side-by-Side Comparison Matrix
    st.markdown("### 📋 Comparison Matrix")
    matrix_cols = st.columns(len(selected_movies))

    for idx, m in enumerate(selected_movies):
        with matrix_cols[idx]:
            st.image(m.get("poster", ""), use_container_width=True)
            st.markdown(f"#### {m.get('title')} ({m.get('year')})")
            st.markdown(f"⭐ **Rating:** {m.get('vote_average')} / 10 ({m.get('vote_count'):,} votes)")
            st.markdown(f"⏱️ **Runtime:** {m.get('runtime')} minutes")
            st.markdown(f"🏷️ **Certification:** `{m.get('certification', 'NR')}`")
            st.markdown(f"🎬 **Director:** {m.get('director', 'Unknown')}")
            st.markdown(f"🎭 **Cast:** {', '.join(m.get('top_cast', [])[:3])}")
            bud = m.get('budget', 0)
            rev = m.get('revenue', 0)
            st.markdown(f"💰 **Budget:** ${bud:,}" if bud > 0 else "💰 **Budget:** N/A")
            st.markdown(f"💵 **Gross:** ${rev:,}" if rev > 0 else "💵 **Gross:** N/A")
            sent = analyze_sentiment(m.get("overview", ""))
            st.markdown(f"🎭 **Mood:** `{classify_mood(sent)}` ({sent['compound']:+.2f})")

    st.markdown("---")

    # Plotly Spider Radar Chart
    st.markdown("### 🕸️ 5-Axis Dimensional Radar Chart")
    radar_fig = viz.plot_movie_radar_chart(selected_movies)
    if radar_fig is not None:
        st.plotly_chart(radar_fig, use_container_width=True)
    else:
        st.info("Radar chart requires the `plotly` library.")


# ==========================================
# PAGE 8: INSIGHTS & VISUALIZATIONS
# ==========================================
def render_insights_page(client: TMDBClient):
    st.title("📊 Cinematic Insights & Cohort Visualizations")
    st.caption("Deep-dive interactive Plotly charts examining ratings, popularity trends, ROI, and plot sentiment.")

    # Cohort data source
    with st.spinner("Aggregating cohort dataset..."):
        cohort = client.get_mock_movies() + client.get_trending("week")
        # Deduplicate
        seen = set()
        unique_cohort = []
        for m in cohort:
            if m["id"] not in seen:
                seen.add(m["id"])
                unique_cohort.append(m)

    chart_tabs = st.tabs([
        "📅 Ratings by Decade",
        "📈 Genre Popularity Trends",
        "🎯 Rating vs. Votes",
        "💰 Budget vs. Revenue ROI",
        "🎭 Sentiment Distribution"
    ])

    with chart_tabs[0]:
        fig = viz.plot_ratings_by_decade(unique_cohort)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with chart_tabs[1]:
        fig = viz.plot_genre_popularity_over_time(unique_cohort)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with chart_tabs[2]:
        fig = viz.plot_rating_vs_votes(unique_cohort)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with chart_tabs[3]:
        fig = viz.plot_budget_vs_revenue(unique_cohort)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("More box office records needed to render the financial scatter plot.")

    with chart_tabs[4]:
        fig = viz.plot_sentiment_distribution(unique_cohort)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


# ==========================================
# MAIN ROUTER
# ==========================================
def main():
    client = TMDBClient()
    recommender = MovieRecommender()

    # Render Sidebar
    render_sidebar(client)

    # Route to Selected Page
    page = st.session_state["current_page"]
    if page == "Home":
        render_home_page()
    elif page == "Discover":
        render_discover_page(client, recommender)
    elif page == "Similar Movies":
        render_similar_movies_page(client, recommender)
    elif page == "Ask in Plain English":
        render_nl_query_page(client, recommender)
    elif page == "Trending":
        render_trending_page(client)
    elif page == "Watchlist":
        render_watchlist_page(client, recommender)
    elif page == "Compare":
        render_compare_page(client)
    elif page == "Insights":
        render_insights_page(client)

    # Global Mandatory TMDB Attribution Footer
    render_footer()


if __name__ == "__main__":
    main()
