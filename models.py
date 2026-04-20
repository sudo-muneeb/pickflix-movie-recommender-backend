from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


# ── Request models ────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    liked_indices: list[int]
    """External TMDB IDs the user has liked."""

    page: int = 0
    exclude_indices: list[int] = []
    """TMDB IDs to suppress from results (already seen / already liked)."""

    # Optional filters (mirrors recommend_from_indices signature)
    same_language: bool = False
    same_era: bool = False
    era_window: int = 15
    min_weighted_rating: float = 5.5
    k: int = 20


# ── Shared movie output ───────────────────────────────────────────────────────

class MovieOut(BaseModel):
    """Compact movie representation returned in lists and search results."""
    movie_index: int
    """External TMDB ID — stable across all endpoints."""
    title: str
    year: Optional[int] = None
    language: Optional[str] = None
    avg_rating: float
    """Exposed as IMDB weighted_rating for all movies (uniform quality signal)."""
    vote_count: int
    poster_path: Optional[str] = None
    score: Optional[float] = None
    """Blended recommendation score (0–1), only present in /recommend responses."""


# ── List / paginated response ─────────────────────────────────────────────────

class MovieListResponse(BaseModel):
    movies: list[MovieOut]
    page: int
    has_more: bool
    pagination_key: Optional[str] = None


# ── Search response ───────────────────────────────────────────────────────────

class SearchResponse(BaseModel):
    results: list[MovieOut]


# ── Discover response ─────────────────────────────────────────────────────────

class DiscoverResponse(BaseModel):
    movies: list[MovieOut]


# ── Full movie detail ─────────────────────────────────────────────────────────

class MovieDetail(BaseModel):
    """All human-facing fields for a single movie."""

    # Identity
    movie_index: int
    title: str
    original_title: Optional[str] = None
    tagline: Optional[str] = None
    overview: Optional[str] = None

    # Release
    release_date: Optional[str] = None
    year: Optional[int] = None
    status: Optional[str] = None
    runtime: Optional[int] = None          # minutes
    adult: bool = False

    # Classification
    language: Optional[str] = None
    genres: Optional[str] = None
    keywords: Optional[str] = None

    # People
    directors: Optional[str] = None
    writers: Optional[str] = None
    cast: Optional[str] = None

    # Ratings — v2.2: weighted_rating is the primary quality signal
    avg_rating: Optional[float] = None     # 8K-only Bayesian average (diagnostic)
    vote_count: Optional[int] = None
    vote_average: Optional[float] = None   # Raw TMDB score
    weighted_rating: Optional[float] = None  # IMDB-formula WR (all 282K movies)
    popularity: Optional[float] = None
    has_real_ratings: bool = False

    # Financials
    budget: Optional[int] = None
    revenue: Optional[int] = None

    # Media
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    homepage: Optional[str] = None

    # Production
    production_companies: Optional[str] = None
    production_countries: Optional[str] = None
    spoken_languages: Optional[str] = None

    # External IDs
    imdb_id: Optional[str] = None         # formatted as "tt0123456"
    tmdb_id: Optional[int] = None
