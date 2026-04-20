"""
main.py — FastAPI application for the Pickflix v2.2 hybrid recommender.

Endpoints
─────────
  GET  /movies/default          Paginated high-quality movies (shuffled per session)
  POST /recommend               Hybrid content + LightFM personalised recommendations
  GET  /movies/search           Fuzzy title search
  GET  /movies/discover         Random sample of high-rated movies (browsing / discovery)
  GET  /movies/{movie_index}    Full metadata for a single movie (TMDB ID)
  GET  /health                  Liveness probe

The recommendation engine is the v2.2 hybrid pipeline from
  notebooks/movie_recommender.ipynb
consisting of:
  1. Content-FAISS retrieval  (multi-query max-fusion)
  2. LightFM collaborative rerank with per-candidate segmented alpha
     (ALPHA_WARM for 8K rated items, ALPHA_COLD for 200K cold-start items)
  3. IMDB weighted_rating quality filter

All movie_index values are TMDB IDs.
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from models import (
    DiscoverResponse,
    MovieDetail,
    MovieListResponse,
    RecommendRequest,
    SearchResponse,
)
import recommender


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pickflix")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Pickflix Movie Recommender API — v2.2",
    description=(
        "Hybrid content + LightFM recommender covering ~282K movies. "
        "Segments cold-start items (200K catalog) from warm items (8K with real ratings) "
        "to avoid letting un-validated LightFM embeddings dominate cold-item ranking."
    ),
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # CRA / fallback
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Logging middleware ────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    qs = f"?{request.url.query}" if request.url.query else ""
    log.info(
        "%s %s%s  →  %d  (%.1f ms)",
        request.method, request.url.path, qs,
        response.status_code, elapsed_ms,
    )
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/movies/default",
    response_model=MovieListResponse,
    tags=["Movies"],
    summary="Paginated high-quality movies",
)
def get_top_movies(
    lang: str | None = Query(
        default=None,
        description="ISO 639-1 language code (e.g. 'en', 'ko', 'hi'). Omit for all languages.",
    ),
    page_offset: int = Query(default=0, description="Page number (0-indexed)."),
    pagination_key: str | None = Query(
        default=None,
        description="Session key returned by a previous request. Omit to start a new session.",
    ),
):
    """
    Returns high-quality movies sorted by IMDB weighted_rating, shuffled
    reproducibly for the current pagination session.

    Movies are drawn from the top 20K by weighted_rating across the full
    282K-movie catalog.
    """
    result = recommender.get_top_movies(
        lang=lang,
        page_offset=page_offset,
        pagination_key=pagination_key,
    )
    titles = [m["title"] for m in result["movies"][:3]]
    key_display = (result.get("pagination_key") or "")[:8]
    log.info(
        "[/movies/default] lang=%s page=%d key=%s → %d movies | first: %s",
        lang or "*", page_offset, key_display, len(result["movies"]), titles,
    )
    return result


@app.post(
    "/recommend",
    response_model=dict,
    tags=["Recommendations"],
    summary="Hybrid personalised recommendations",
)
def recommend(body: RecommendRequest):
    """
    Return up to `k` personalised movie recommendations using the v2.2 hybrid pipeline:

    1. Content-FAISS retrieval across all 282K movies (multi-query max-fusion union).
    2. LightFM collaborative reranking with **per-candidate segmented alpha**:
       - `ALPHA_WARM` for movies with real user ratings (8K catalog).
       - `ALPHA_COLD` for cold-start 200K-only movies (content-heavy to avoid
         trusting un-validated LightFM embeddings).
    3. IMDB `weighted_rating` quality filter.

    `liked_indices` — TMDB IDs of movies the user liked (≥ 1 required).
    `exclude_indices` — TMDB IDs to suppress (already shown / liked).
    """
    if not body.liked_indices:
        raise HTTPException(
            status_code=422,
            detail="`liked_indices` must contain at least one TMDB movie ID.",
        )

    # Validate that every liked ID exists in the catalog
    missing = [tid for tid in body.liked_indices if tid not in recommender._tmdb_to_row]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"TMDB IDs not found in catalog: {missing}",
        )

    result = recommender.get_recommendations(
        liked_indices=body.liked_indices,
        exclude_indices=body.exclude_indices,
        k=body.k,
        min_weighted_rating=body.min_weighted_rating,
        same_language=body.same_language,
        same_era=body.same_era,
        era_window=body.era_window,
    )
    titles = [m["title"] for m in result["movies"][:3]]
    log.info(
        "[/recommend] liked=%s → %d recs | first: %s",
        body.liked_indices, len(result["movies"]), titles,
    )
    return result


@app.get(
    "/movies/search",
    response_model=SearchResponse,
    tags=["Movies"],
    summary="Fuzzy title search",
)
def search_movies(
    q: str = Query(..., min_length=1, description="Search query (partial or full movie title)."),
):
    """
    Fuzzy-match `q` against all ~282K movie titles.

    Scoring combines token overlap, partial-ratio fuzzy matching, and
    popularity, with exact-match and prefix-match boosts.
    Returns up to 25 results.
    """
    result = recommender.search_movies(query=q)
    titles = [m["title"] for m in result["results"][:3]]
    log.info(
        "[/movies/search] q=%r → %d hits | first: %s",
        q, len(result["results"]), titles,
    )
    return result


@app.get(
    "/movies/discover",
    response_model=DiscoverResponse,
    tags=["Movies"],
    summary="Random sample of high-quality movies for browsing / discovery",
)
def discover_movies(
    n: int = Query(default=20, ge=1, le=100, description="Number of movies to return."),
    min_weighted_rating: float = Query(
        default=6.5,
        description="Minimum IMDB weighted_rating (0–10). Default 6.5 surfaces well-regarded films.",
    ),
    language: str | None = Query(
        default=None,
        description="ISO 639-1 language code filter (e.g. 'hi', 'ko', 'fr').",
    ),
    year_from: int | None = Query(default=None, description="Earliest release year (inclusive)."),
    year_to: int | None = Query(default=None, description="Latest release year (inclusive)."),
    seed: int | None = Query(default=None, description="RNG seed for reproducible samples."),
):
    """
    Returns a random sample of high-quality movies, optionally filtered by
    language and release year range. Mirrors `show_high_rated_random()` from
    the notebook.
    """
    result = recommender.get_discover_movies(
        n=n,
        min_weighted_rating=min_weighted_rating,
        language=language,
        year_from=year_from,
        year_to=year_to,
        seed=seed,
    )
    log.info(
        "[/movies/discover] lang=%s yr=%s-%s wr>=%s → %d movies",
        language or "*", year_from or "?", year_to or "?",
        min_weighted_rating, len(result["movies"]),
    )
    return result


@app.get(
    "/movies/{movie_index}",
    response_model=MovieDetail,
    tags=["Movies"],
    summary="Full metadata for a single movie",
)
def get_movie_detail(movie_index: int):
    """
    Return every available field for the movie with the given TMDB ID.

    Raises **404** if the TMDB ID is not in the catalog.
    """
    if movie_index not in recommender._tmdb_to_row:
        raise HTTPException(
            status_code=404,
            detail=f"TMDB ID {movie_index} not found in catalog.",
        )
    detail = recommender.get_movie_detail(movie_index)
    log.info(
        "[/movies/%d] → %s (%s)",
        movie_index, detail.get("title"), detail.get("year"),
    )
    return detail


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["Infra"], summary="Liveness probe")
def health():
    """Returns 200 + catalog stats when the server is ready."""
    return {
        "status":       "ok",
        "version":      "2.2.0",
        "movies_loaded": len(recommender.all_movies),
        "lightfm_on":   recommender._use_lightfm,
        "alpha_warm":   recommender._ALPHA_WARM,
        "alpha_cold":   recommender._ALPHA_COLD,
        "min_wr":       recommender._MIN_WR,
    }
