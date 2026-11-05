"""
viz.py - Interactive Plotly Visualizations with Dark Cinematic Theme

Visualizations Implemented:
1. Ratings by Decade (Box Plot)
2. Genre Popularity Over Time (Multi-line / Trend Chart)
3. Rating vs. Vote Count (Bubble Scatter with Popularity scaling)
4. Budget vs. Box Office Revenue (Scatter with Break-Even Diagonal & ROI metrics)
5. Sentiment Distribution (Histogram colored by Mood classification)
6. Multi-Movie Radar / Spider Chart (Normalized 5-Axis comparison for 2-3 titles)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from collections import defaultdict

try:
    import plotly.graph_objects as go
    import plotly.express as px
    _PLOTLY_AVAILABLE = True
except ImportError:
    go = None  # type: ignore
    px = None  # type: ignore
    _PLOTLY_AVAILABLE = False

from nlp import analyze_sentiment, classify_mood

logger = logging.getLogger(__name__)

# Netflix-Inspired Theme Constants
THEME_BG = "#141414"
THEME_PAPER = "#1F1F1F"
THEME_TEXT = "#FFFFFF"
THEME_GRID = "rgba(255, 255, 255, 0.08)"
PALETTE = ["#E50914", "#46D369", "#FFA000", "#00B0FF", "#B81D24", "#FFFFFF"]


def apply_cinematic_layout(fig: Any, title: str, height: int = 450) -> Any:
    """Applies consistent Netflix-style dark layout, bold typography, and clean grid lines to a Plotly figure."""
    if not _PLOTLY_AVAILABLE or fig is None:
        return None

    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "font": {"size": 17, "color": "#FFFFFF", "family": "Inter, -apple-system, sans-serif"},
            "x": 0.02,
            "y": 0.95,
        },
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_PAPER,
        font={"color": THEME_TEXT, "family": "Inter, -apple-system, sans-serif"},
        height=height,
        margin={"l": 50, "r": 30, "t": 60, "b": 50},
        legend={"bgcolor": "rgba(31, 31, 31, 0.85)", "font": {"color": THEME_TEXT}},
        xaxis={"gridcolor": THEME_GRID, "linecolor": THEME_GRID, "zerolinecolor": THEME_GRID},
        yaxis={"gridcolor": THEME_GRID, "linecolor": THEME_GRID, "zerolinecolor": THEME_GRID},
    )
    return fig


def plot_ratings_by_decade(movies: List[Dict[str, Any]]) -> Optional[Any]:
    """Box plot of movie ratings grouped by release decade."""
    if not _PLOTLY_AVAILABLE or not movies:
        return None

    data_by_decade: Dict[str, List[float]] = defaultdict(list)
    for m in movies:
        year = m.get("year", 0)
        rating = float(m.get("vote_average", 0.0))
        if year >= 1950 and rating > 0:
            dec = f"{(year // 10) * 10}s"
            data_by_decade[dec].append(rating)

    if not data_by_decade:
        return None

    sorted_decades = sorted(data_by_decade.keys())

    fig = go.Figure()
    for i, dec in enumerate(sorted_decades):
        fig.add_trace(go.Box(
            y=data_by_decade[dec],
            name=dec,
            marker_color=PALETTE[i % len(PALETTE)],
            boxmean=True,
            boxpoints="outliers",
        ))

    fig.update_layout(
        xaxis_title="Decade",
        yaxis_title="TMDB Rating (0–10)",
        yaxis={"range": [0, 10]},
        showlegend=False,
    )
    return apply_cinematic_layout(fig, "Rating Distribution Across Decades")


def plot_genre_popularity_over_time(movies: List[Dict[str, Any]]) -> Optional[Any]:
    """Trend line showing average movie popularity per decade for top genres."""
    if not _PLOTLY_AVAILABLE or not movies:
        return None

    # Aggregate popularity by (decade, genre)
    genre_decade_pop: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    genre_counts: Dict[str, int] = defaultdict(int)

    for m in movies:
        year = m.get("year", 0)
        pop = float(m.get("popularity", 0.0))
        if year >= 1960:
            dec = f"{(year // 10) * 10}s"
            for g in m.get("genres", []):
                genre_decade_pop[g][dec].append(pop)
                genre_counts[g] += 1

    top_genres = [g for g, _ in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
    if not top_genres:
        return None

    all_decades = sorted(list(set(
        dec for g in top_genres for dec in genre_decade_pop[g].keys()
    )))

    fig = go.Figure()
    for i, g in enumerate(top_genres):
        x_vals = []
        y_vals = []
        for dec in all_decades:
            scores = genre_decade_pop[g].get(dec, [])
            if scores:
                x_vals.append(dec)
                y_vals.append(round(sum(scores) / len(scores), 1))

        if x_vals:
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name=g,
                line={"width": 3, "color": PALETTE[i % len(PALETTE)]},
                marker={"size": 8},
            ))

    fig.update_layout(
        xaxis_title="Decade",
        yaxis_title="Average Popularity Index",
    )
    return apply_cinematic_layout(fig, "Top Genres Popularity Trends Over Time")


def plot_rating_vs_votes(movies: List[Dict[str, Any]]) -> Optional[Any]:
    """Scatter plot of vote count vs. average rating with point size scaled by popularity."""
    if not _PLOTLY_AVAILABLE or not movies:
        return None

    valid = [m for m in movies if m.get("vote_count", 0) > 0 and m.get("vote_average", 0) > 0]
    if not valid:
        return None

    titles = [m.get("title", "") for m in valid]
    ratings = [float(m.get("vote_average", 0.0)) for m in valid]
    votes = [int(m.get("vote_count", 0)) for m in valid]
    pops = [float(m.get("popularity", 10.0)) for m in valid]
    genres = [", ".join(m.get("genres", [])[:2]) for m in valid]

    # Normalize bubble sizes
    max_pop = max(pops) if pops else 1.0
    sizes = [max(8, min(30, int((p / max_pop) * 30))) for p in pops]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=votes,
        y=ratings,
        mode="markers",
        text=[f"<b>{t}</b><br>Rating: {r}★<br>Votes: {v:,}<br>Genres: {g}" for t, r, v, g in zip(titles, ratings, votes, genres)],
        hoverinfo="text",
        marker={
            "size": sizes,
            "color": ratings,
            "colorscale": "Inferno",
            "showscale": True,
            "colorbar": {"title": "Rating", "tickfont": {"color": THEME_TEXT}},
            "line": {"width": 1, "color": "#FFFFFF"},
        }
    ))

    fig.update_layout(
        xaxis_title="Number of User Votes (Log Scale)",
        xaxis_type="log",
        yaxis_title="Average Rating (0–10)",
        yaxis={"range": [4.0, 10.0]},
    )
    return apply_cinematic_layout(fig, "Quality vs. Reach: Rating vs. Total Votes")


def plot_budget_vs_revenue(movies: List[Dict[str, Any]]) -> Optional[Any]:
    """Scatter plot comparing production budget against box office gross with break-even line."""
    if not _PLOTLY_AVAILABLE or not movies:
        return None

    # Filter films with recorded budget & revenue
    valid = [
        m for m in movies
        if int(m.get("budget", 0)) > 1000000 and int(m.get("revenue", 0)) > 1000000
    ]
    if not valid:
        return None

    titles = [m.get("title", "") for m in valid]
    budgets_m = [int(m.get("budget", 0)) / 1_000_000 for m in valid]
    revenues_m = [int(m.get("revenue", 0)) / 1_000_000 for m in valid]
    rois = [round(((rev - bud) / bud) * 100, 1) for bud, rev in zip(budgets_m, revenues_m)]

    max_val = max(max(budgets_m), max(revenues_m)) * 1.1

    fig = go.Figure()

    # 1:1 Break-Even Reference Line
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode="lines",
        name="Break-Even (1x)",
        line={"dash": "dash", "color": "#718096", "width": 1.5},
    ))

    # Movies scatter
    fig.add_trace(go.Scatter(
        x=budgets_m,
        y=revenues_m,
        mode="markers+text",
        text=titles,
        textposition="top center",
        hovertext=[
            f"<b>{t}</b><br>Budget: ${b:.1f}M<br>Gross: ${r:.1f}M<br>ROI: {roi:+,}%"
            for t, b, r, roi in zip(titles, budgets_m, revenues_m, rois)
        ],
        hoverinfo="text",
        marker={
            "size": 12,
            "color": rois,
            "colorscale": "Viridis",
            "showscale": True,
            "colorbar": {"title": "ROI (%)"},
            "line": {"width": 1, "color": "#FFFFFF"},
        }
    ))

    fig.update_layout(
        xaxis_title="Production Budget ($ Millions)",
        yaxis_title="Worldwide Gross Revenue ($ Millions)",
    )
    return apply_cinematic_layout(fig, "Financial Returns: Budget vs. Box Office Gross")


def plot_sentiment_distribution(movies: List[Dict[str, Any]]) -> Optional[Any]:
    """Histogram of VADER sentiment compound scores colored by mood archetypes."""
    if not _PLOTLY_AVAILABLE or not movies:
        return None

    mood_counts: Dict[str, int] = defaultdict(int)
    compound_scores: List[float] = []

    for m in movies:
        overview = m.get("overview", "")
        sent = analyze_sentiment(overview)
        mood = classify_mood(sent)
        mood_counts[mood] += 1
        compound_scores.append(sent["compound"])

    if not compound_scores:
        return None

    mood_colors = {
        "Uplifting": "#46D369",
        "Light-hearted": "#FFA000",
        "Neutral": "#A3A3A3",
        "Dark": "#E50914",
        "Intense": "#9C27B0",
    }

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(mood_counts.keys()),
        y=list(mood_counts.values()),
        marker_color=[mood_colors.get(k, "#FFFFFF") for k in mood_counts.keys()],
        text=list(mood_counts.values()),
        textposition="auto",
    ))

    fig.update_layout(
        xaxis_title="Detected Plot Mood Profile",
        yaxis_title="Movie Count",
        showlegend=False,
    )
    return apply_cinematic_layout(fig, "Emotional Landscape: Narrative Mood Breakdown")


def plot_movie_radar_chart(movies: List[Dict[str, Any]]) -> Optional[Any]:
    """
    Radar chart comparing 2-3 movies across 5 normalized dimensions (0 to 100):
      1. Rating
      2. Popularity
      3. Runtime
      4. Revenue
      5. Sentiment Positivity
    """
    if not _PLOTLY_AVAILABLE or not movies or len(movies) < 2:
        return None

    categories = [
        "Rating (Score)",
        "Popularity",
        "Runtime (Pacing)",
        "Box Office Revenue",
        "Sentiment (Positivity)",
    ]

    # Precompute maximum values for normalizing
    max_pop = max([float(m.get("popularity", 1.0)) for m in movies] + [100.0])
    max_rev = max([int(m.get("revenue", 1)) for m in movies] + [100_000_000])

    fig = go.Figure()

    for i, m in enumerate(movies[:3]):
        rating_norm = min(100.0, float(m.get("vote_average", 0.0)) * 10.0)
        pop_norm = min(100.0, (float(m.get("popularity", 0.0)) / max_pop) * 100.0)
        runtime_norm = min(100.0, (float(m.get("runtime", 100)) / 180.0) * 100.0)
        rev_norm = min(100.0, (float(m.get("revenue", 0)) / max_rev) * 100.0)

        sent = analyze_sentiment(m.get("overview", ""))
        sent_norm = min(100.0, max(0.0, ((sent["compound"] + 1.0) / 2.0) * 100.0))

        values = [rating_norm, pop_norm, runtime_norm, rev_norm, sent_norm]
        # Close loop for radar
        values.append(values[0])
        radar_categories = categories + [categories[0]]

        color = PALETTE[i % len(PALETTE)]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_categories,
            fill="toself",
            name=m.get("title", f"Movie {i+1}"),
            line={"color": color, "width": 2},
            opacity=0.6,
        ))

    fig.update_layout(
        polar={
            "bgcolor": THEME_PAPER,
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
                "linecolor": THEME_GRID,
                "gridcolor": THEME_GRID,
                "tickfont": {"color": THEME_TEXT},
            },
            "angularaxis": {
                "linecolor": THEME_GRID,
                "gridcolor": THEME_GRID,
                "tickfont": {"color": "#FFFFFF", "size": 12},
            },
        },
        showlegend=True,
    )
    return apply_cinematic_layout(fig, "Side-by-Side Dimensional Comparison", height=500)
