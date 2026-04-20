"""
recommender.py — v2.2 Hybrid Recommender backend module.

Implements the same multi-signal hybrid pipeline defined in
notebooks/movie_recommender.ipynb (v2.2):

  Stage 1  Content-FAISS retrieval   — finds candidate pool for all 282K movies
  Stage 2  LightFM collaborative rerank — per-candidate segmented alpha blend
  Stage 3  Quality filter             — IMDB weighted_rating >= PRODUCTION_MIN_WR

Artifacts produced by the notebook (section 21) that this module expects:

  artifacts/all_movies.parquet          — merged 282K movie catalog
  artifacts/all_content_embeddings.npy  — L2-normalised (N, 300) content matrix
  artifacts/content_index.faiss         — IndexFlatIP over all_content_embeddings
  artifacts/lightfm_item_embeddings.npy — (N, 64) LightFM production embeddings
  artifacts/lightfm_item_biases.npy     — (N,) LightFM item biases
  artifacts/blend_config.json           — ALPHA_WARM, ALPHA_COLD, PRODUCTION_MIN_WR, …

All five are consumed here; the TF-IDF / SVD / scaler pickles are not needed
at serving time (they are only used when transforming a *new* unseen movie at
query time, which is not part of the current serving path).
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import uuid
from collections import defaultdict
from typing import Optional

import faiss
import numpy as np
import pandas as pd

# ── Path resolution ───────────────────────────────────────────────────────────
_HERE = pathlib.Path(__file__).parent
_ARTIFACTS = _HERE / "artifacts"


# ── Artifact loading ──────────────────────────────────────────────────────────

def _load_artifacts():
    """Load all v2.2 artifacts from disk and return them."""

    print("[recommender] Loading all_movies.parquet …")
    movies = pd.read_parquet(_ARTIFACTS / "all_movies.parquet")
    # Reset to a clean integer RangeIndex so iloc works correctly with FAISS positions.
    # We keep the original 'id' (TMDB ID) column for external lookups.
    movies = movies.reset_index(drop=True)

    print("[recommender] Loading all_content_embeddings.npy …")
    embeddings = np.load(
        _ARTIFACTS / "all_content_embeddings.npy"
    ).astype(np.float32)

    print("[recommender] Loading content_index.faiss …")
    content_index = faiss.read_index(str(_ARTIFACTS / "content_index.faiss"))

    # LightFM artifacts are optional — the system degrades to content-only if absent.
    lf_embeddings: Optional[np.ndarray] = None
    lf_biases: Optional[np.ndarray] = None
    use_lightfm = False

    lf_emb_path = _ARTIFACTS / "lightfm_item_embeddings.npy"
    lf_bias_path = _ARTIFACTS / "lightfm_item_biases.npy"
    if lf_emb_path.exists() and lf_bias_path.exists():
        print("[recommender] Loading lightfm_item_embeddings.npy + lightfm_item_biases.npy …")
        lf_embeddings = np.load(lf_emb_path).astype(np.float32)
        lf_biases = np.load(lf_bias_path).astype(np.float32)
        use_lightfm = True
    else:
        print("[recommender] LightFM artifacts not found — using content-only scoring.")

    # Blend / quality config
    blend_config: dict = {
        "ALPHA_WARM": 0.1,
        "ALPHA_COLD": 0.8,
        "LIKE_THRESHOLD": 4.0,
        "PRODUCTION_MIN_WR": 5.5,
    }
    blend_path = _ARTIFACTS / "blend_config.json"
    if blend_path.exists():
        with open(blend_path) as f:
            blend_config.update(json.load(f))

    print(
        f"[recommender] Ready — {len(movies):,} movies, "
        f"embeddings {embeddings.shape}, index ntotal={content_index.ntotal}, "
        f"lightfm={'on' if use_lightfm else 'off'}, "
        f"ALPHA_WARM={blend_config['ALPHA_WARM']}, "
        f"ALPHA_COLD={blend_config['ALPHA_COLD']}, "
        f"MIN_WR={blend_config['PRODUCTION_MIN_WR']}"
    )

    return movies, embeddings, content_index, lf_embeddings, lf_biases, use_lightfm, blend_config


# ── Module-level singletons ───────────────────────────────────────────────────

(
    all_movies,
    all_content_emb,
    content_index,
    _lf_embeddings,
    _lf_biases,
    _use_lightfm,
    _blend_cfg,
) = _load_artifacts()

# Convenience aliases derived from blend_config
_ALPHA_WARM: float = _blend_cfg["ALPHA_WARM"]
_ALPHA_COLD: float = _blend_cfg["ALPHA_COLD"]
_MIN_WR: float = _blend_cfg["PRODUCTION_MIN_WR"]

# Build TMDB-ID → row index mapping (row index == FAISS position)
# 'id' is the TMDB/IMDB ID stored in the parquet; use it as the external key.
_id_col = "id" if "id" in all_movies.columns else all_movies.columns[0]
_tmdb_to_row: dict[int, int] = {
    int(mid): pos
    for pos, mid in enumerate(all_movies[_id_col])
    if pd.notna(mid)
}
# Reverse: FAISS position → TMDB ID
_row_to_tmdb: np.ndarray = all_movies[_id_col].fillna(0).astype(np.int64).values


# ── Pagination session cache ──────────────────────────────────────────────────
_pagination_sessions: dict[str, dict] = {}

# ── Top-movies cache ──────────────────────────────────────────────────────────
_top_movies_cache: Optional[pd.DataFrame] = None


# ── Core helpers ──────────────────────────────────────────────────────────────

def _minmax_norm(x: np.ndarray) -> np.ndarray:
    """Per-query min-max normalisation — identical to notebook's `minmax_norm()`."""
    x = np.asarray(x, dtype=np.float64)
    span = x.max() - x.min()
    if span < 1e-9:
        return np.zeros_like(x)
    return (x - x.min()) / span


def _segment_alpha(candidate_row_indices: np.ndarray) -> np.ndarray:
    """
    Per-candidate alpha — mirrors notebook `segment_alpha()`.
    Movies with `has_real_ratings=True` use ALPHA_WARM (trust LightFM more);
    cold-start 200K-only movies use ALPHA_COLD (lean on content).
    """
    has_ratings = all_movies["has_real_ratings"].iloc[candidate_row_indices].to_numpy()
    return np.where(has_ratings, _ALPHA_WARM, _ALPHA_COLD)


def _safe_str(v) -> Optional[str]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return s if s else None


def _safe_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (ValueError, TypeError):
        return None


def _row_to_dict(row: pd.Series, score: Optional[float] = None) -> dict:
    """Convert a DataFrame row → plain dict matching MovieOut schema."""

    # year: prefer release_year (computed in notebook section 7), else release_date
    year: Optional[int] = None
    if "release_year" in row.index:
        year = _safe_int(row["release_year"])
    if year is None:
        rd = _safe_str(row.get("release_date"))
        if rd and len(rd) >= 4:
            try:
                year = int(rd[:4])
            except ValueError:
                pass

    lang = row.get("original_language")
    lang = lang if pd.notna(lang) else None  # type: ignore[arg-type]

    # weighted_rating is the new quality signal; avg_rating is 8K-only diagnostics.
    # Expose weighted_rating as avg_rating for front-end compatibility.
    wr = _safe_float(row.get("weighted_rating")) or 0.0
    avg = _safe_float(row.get("avg_rating")) or wr  # fall back to WR if no avg

    vote_count_raw = row.get("vote_count", 0)
    # vote_count was log1p-transformed during preprocessing — undo it for display
    vote_count = int(round(math.expm1(float(vote_count_raw)))) if pd.notna(vote_count_raw) else 0

    poster = row.get("poster_path")
    poster = poster if pd.notna(poster) else None  # type: ignore[arg-type]

    return {
        "movie_index": int(row[_id_col]),   # external TMDB ID
        "title":       str(row["title"]),
        "year":        year,
        "language":    lang,
        "avg_rating":  round(wr, 2),        # expose WR as avg_rating
        "vote_count":  vote_count,
        "poster_path": poster,
        "score":       round(float(score), 4) if score is not None else None,
    }


# ── Top movies ────────────────────────────────────────────────────────────────

def _get_top_movies_cached() -> pd.DataFrame:
    """Cache the highest-quality movies (by IMDB weighted_rating) once at startup."""
    global _top_movies_cache
    if _top_movies_cache is not None:
        return _top_movies_cache

    print("[recommender] Building top-movies cache …")
    df = all_movies.copy()

    # Require a minimum vote_count (post-log1p transform ≥ log1p(50) ≈ 3.93)
    min_vc_log = math.log1p(50)
    df = df[df["vote_count"].fillna(0) >= min_vc_log]

    # Sort by IMDB weighted_rating desc, then vote_count desc
    sort_cols = [c for c in ["weighted_rating", "vote_count"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=False)

    df = df.head(20_000)
    _top_movies_cache = df
    print(f"[recommender] Top-movies cache ready — {len(df):,} movies")
    return df


def get_top_movies(
    pagination_key: Optional[str] = None,
    lang: Optional[str] = None,
    page_offset: int = 0,
    page_size: int = 20,
) -> dict:
    """
    Return paginated high-quality movies, shuffled reproducibly per session.

    `lang`  — ISO 639-1 language code filter; None = all languages.
    `page_offset` — 0-indexed page number.
    """
    if pagination_key is None:
        pagination_key = str(uuid.uuid4())

    if pagination_key not in _pagination_sessions:
        df = _get_top_movies_cached().copy()

        if lang:
            df = df[df["original_language"] == lang]

        seed = hash(pagination_key) % (2 ** 31)
        df = df.sample(frac=1, random_state=seed)

        _pagination_sessions[pagination_key] = {
            "lang": lang,
            "shuffled_df": df,
        }

    session = _pagination_sessions[pagination_key]
    df = session["shuffled_df"]

    total = len(df)
    start = page_offset * page_size
    end = start + page_size
    batch = df.iloc[start:end]

    movies = [_row_to_dict(row) for _, row in batch.iterrows()]
    return {
        "movies":         movies,
        "page":           page_offset,
        "has_more":       end < total,
        "pagination_key": pagination_key,
        "sort_method":    "weighted_rating desc, shuffled by pagination_key",
    }


# ── Core hybrid recommender ───────────────────────────────────────────────────

def get_recommendations(
    liked_indices: list[int],        # external TMDB IDs
    exclude_indices: list[int],      # TMDB IDs already shown / liked
    k: int = 20,
    min_weighted_rating: float = _MIN_WR,
    same_language: bool = False,
    same_era: bool = False,
    era_window: int = 15,
    candidate_multiplier: int = 25,
) -> dict:
    """
    Hybrid recommender — mirrors notebook `recommend_from_indices()`:

    1. Per-liked-item FAISS retrieval with max-fusion union.
    2. LightFM collaborative rerank (segmented alpha; content-only fallback).
    3. Quality filter on IMDB weighted_rating.
    4. Optional language / era filters.
    """
    # Map external TMDB IDs → internal row / FAISS positions
    liked_rows = [_tmdb_to_row[tid] for tid in liked_indices if tid in _tmdb_to_row]
    if not liked_rows:
        return {"movies": [], "has_more": False}

    exclude_set: set[int] = set(exclude_indices) | set(liked_indices)

    # Input metadata for optional filters
    input_df = all_movies.iloc[liked_rows]
    input_langs = set(input_df["original_language"].dropna().tolist())
    input_years = input_df["release_year"].dropna().tolist() if "release_year" in input_df.columns else []
    year_mean = float(np.mean(input_years)) if input_years else None

    # ── Stage 1: per-item FAISS retrieval with max-score fusion ──────────────
    n_candidates = k * candidate_multiplier
    per_item_k = max(n_candidates // max(len(liked_rows), 1), 50)

    candidate_pool: dict[int, float] = {}   # row_idx → best content score
    for qi in liked_rows:
        qv = all_content_emb[[qi]].copy().astype(np.float32)
        faiss.normalize_L2(qv)
        scores, idxs = content_index.search(qv, per_item_k)
        for s, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            if i not in candidate_pool or s > candidate_pool[i]:
                candidate_pool[i] = float(s)

    if not candidate_pool:
        return {"movies": [], "has_more": False}

    candidate_idx = np.array(list(candidate_pool.keys()), dtype=np.int64)
    content_scores = np.array([candidate_pool[i] for i in candidate_idx], dtype=np.float64)

    # ── Stage 2: LightFM collaborative rerank ────────────────────────────────
    if _use_lightfm and _lf_embeddings is not None and _lf_biases is not None:
        lf_query = _lf_embeddings[liked_rows].mean(axis=0)          # (64,)
        lf_raw = (
            _lf_biases[candidate_idx]
            + _lf_embeddings[candidate_idx] @ lf_query              # (N,)
        )
        content_norm = _minmax_norm(content_scores)
        lf_norm = _minmax_norm(lf_raw)
        alpha_vec = _segment_alpha(candidate_idx)                    # per-candidate
        blended = alpha_vec * content_norm + (1.0 - alpha_vec) * lf_norm
    else:
        content_norm = _minmax_norm(content_scores)
        lf_norm = np.zeros_like(content_norm)
        alpha_vec = np.ones_like(content_norm)                       # pure content
        blended = content_norm

    # ── Stage 3: rank + filter ────────────────────────────────────────────────
    order = np.argsort(-blended)

    results = []
    for rank_pos in order:
        row_idx = int(candidate_idx[rank_pos])
        row = all_movies.iloc[row_idx]

        # Skip input / already-seen movies
        tmdb_id = int(row[_id_col]) if pd.notna(row[_id_col]) else -1
        if tmdb_id in exclude_set or row_idx in liked_rows:
            continue

        # Quality gate (IMDB weighted_rating)
        wr = float(row.get("weighted_rating", 0) or 0)
        if wr < min_weighted_rating:
            continue

        # Language filter
        if same_language and row.get("original_language") not in input_langs:
            continue

        # Era filter
        if same_era and year_mean is not None and "release_year" in row.index:
            yr = row.get("release_year")
            if yr is None or pd.isna(yr) or abs(float(yr) - year_mean) > era_window:
                continue

        d = _row_to_dict(row, score=float(blended[rank_pos]))
        # Expose the full scoring breakdown for transparency / debugging
        d["content_score"]  = round(float(content_norm[rank_pos]), 4)
        d["lightfm_score"]  = round(float(lf_norm[rank_pos]), 4)
        d["alpha_used"]     = round(float(alpha_vec[rank_pos]), 2)
        d["blended_score"]  = round(float(blended[rank_pos]), 4)
        d["source"]         = "8k" if bool(row.get("has_real_ratings", False)) else "200k"

        results.append(d)
        if len(results) == k:
            break

    return {
        "movies":   results,
        "has_more": len(results) == k,
    }


# ── Search ────────────────────────────────────────────────────────────────────

def search_movies(query: str, n: int = 25) -> dict:
    """
    Fuzzy title search with token + partial-ratio scoring and popularity boost.
    Uses rapidfuzz if available, falls back to difflib.
    """
    if not query or not query.strip():
        return {"results": []}

    q_norm = re.sub(r"[^\w\s]", "", query.lower()).strip()
    q_tokens = set(q_norm.split())

    # Normalise titles once (stored if not already present)
    if "title_norm" not in all_movies.columns:
        all_movies["title_norm"] = (
            all_movies["title"]
            .str.lower()
            .str.replace(r"[^\w\s]", "", regex=True)
        )

    titles_norm = all_movies["title_norm"]

    # Token overlap score
    def _token(t: str) -> float:
        t_tok = set(t.split())
        return len(q_tokens & t_tok) / len(q_tokens) if q_tokens else 0.0

    token_scores = titles_norm.apply(_token)

    # Fuzzy score
    try:
        from rapidfuzz import fuzz as _fuzz
        fuzzy_scores = titles_norm.apply(lambda t: _fuzz.partial_ratio(q_norm, t) / 100.0)
    except ImportError:
        from difflib import SequenceMatcher
        fuzzy_scores = titles_norm.apply(
            lambda t: SequenceMatcher(None, q_norm, t).ratio()
        )

    # Popularity signal (vote_count already log1p-transformed; normalise to [0,1])
    vc = all_movies["vote_count"].fillna(0)
    pop_norm = vc / (vc.max() + 1)

    exact_boost  = (titles_norm == q_norm).astype(float)
    prefix_boost = titles_norm.str.startswith(q_norm).astype(float)

    final_score = (
        token_scores  * 0.45
        + fuzzy_scores * 0.30
        + pop_norm     * 0.15
        + exact_boost  * 1.00
        + prefix_boost * 0.50
    )

    # Keep the most relevant candidates
    top_k = max(n * 3, int(len(all_movies) * 0.02))
    top_idx = final_score.nlargest(top_k).index

    seen: set[str] = set()
    results: list[dict] = []

    for idx in top_idx:
        row = all_movies.loc[idx]
        title = str(row["title"])
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        results.append(_row_to_dict(row))
        if len(results) >= n and final_score[idx] < 0.25:
            break

    return {"results": results}


# ── Discovery ─────────────────────────────────────────────────────────────────

def get_discover_movies(
    n: int = 20,
    min_weighted_rating: float = 6.5,
    language: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    seed: Optional[int] = None,
) -> dict:
    """
    Random sample of high-quality movies — mirrors notebook `show_high_rated_random()`.

    Useful for browsing / discovery UX.
    """
    df = all_movies.copy()
    df = df[df["weighted_rating"].fillna(0) >= min_weighted_rating]

    if language:
        df = df[df["original_language"] == language]
    if year_from and "release_year" in df.columns:
        df = df[df["release_year"].fillna(0) >= year_from]
    if year_to and "release_year" in df.columns:
        df = df[df["release_year"].fillna(9999) <= year_to]

    if df.empty:
        return {"movies": []}

    rng = np.random.default_rng(seed)
    chosen = df.sample(n=min(n, len(df)), random_state=int(rng.integers(0, 2**31)))
    movies = [_row_to_dict(row) for _, row in chosen.iterrows()]
    return {"movies": movies}


# ── Movie detail ──────────────────────────────────────────────────────────────

def get_movie_detail(tmdb_id: int) -> dict:
    """
    Return a clean, human-facing dict for the given TMDB ID.
    Matches the MovieDetail Pydantic schema.
    """
    if tmdb_id not in _tmdb_to_row:
        raise KeyError(tmdb_id)

    row = all_movies.iloc[_tmdb_to_row[tmdb_id]]

    year: Optional[int] = None
    if "release_year" in row.index:
        year = _safe_int(row["release_year"])
    if year is None:
        rd = _safe_str(row.get("release_date"))
        if rd and len(rd) >= 4:
            try:
                year = int(rd[:4])
            except ValueError:
                pass

    imdb_id: Optional[str] = None
    raw_imdb = _safe_int(row.get("imdbId"))
    if raw_imdb is not None:
        imdb_id = f"tt{raw_imdb:07d}"

    # vote_count is log1p-transformed; undo for display
    vc_raw = row.get("vote_count", 0)
    vote_count_display = int(round(math.expm1(float(vc_raw)))) if pd.notna(vc_raw) else None

    # Same for revenue / popularity if stored log-transformed
    rev_raw = row.get("revenue", 0)
    revenue_display = int(round(math.expm1(float(rev_raw)))) if pd.notna(rev_raw) and float(rev_raw) > 0 else _safe_int(rev_raw)

    pop_raw = row.get("popularity", 0)
    popularity_display = _safe_float(math.expm1(float(pop_raw))) if pd.notna(pop_raw) and float(pop_raw) > 0 else _safe_float(pop_raw)

    return {
        # Identity
        "movie_index":          tmdb_id,
        "title":                _safe_str(row.get("title")) or "",
        "original_title":       _safe_str(row.get("original_title")),
        "tagline":              _safe_str(row.get("tagline")),
        "overview":             _safe_str(row.get("overview")),
        # Release
        "release_date":         _safe_str(row.get("release_date")),
        "year":                 year,
        "status":               _safe_str(row.get("status")),
        "runtime":              _safe_int(row.get("runtime")),
        "adult":                bool(row.get("adult", False)),
        # Classification
        "language":             _safe_str(row.get("original_language")),
        "genres":               _safe_str(row.get("genres_y")),
        "keywords":             _safe_str(row.get("keywords")),
        # People
        "directors":            _safe_str(row.get("directors")),
        "writers":              _safe_str(row.get("writers")),
        "cast":                 _safe_str(row.get("cast")),
        # Ratings
        "avg_rating":           _safe_float(row.get("avg_rating")),
        "vote_count":           vote_count_display,
        "vote_average":         _safe_float(row.get("vote_average")),
        "weighted_rating":      _safe_float(row.get("weighted_rating")),
        "popularity":           popularity_display,
        "has_real_ratings":     bool(row.get("has_real_ratings", False)),
        # Financials
        "budget":               _safe_int(row.get("budget")),
        "revenue":              revenue_display,
        # Media
        "poster_path":          _safe_str(row.get("poster_path")),
        "backdrop_path":        _safe_str(row.get("backdrop_path")),
        "homepage":             _safe_str(row.get("homepage")),
        # Production
        "production_companies": _safe_str(row.get("production_companies")),
        "production_countries": _safe_str(row.get("production_countries")),
        "spoken_languages":     _safe_str(row.get("spoken_languages")),
        # External IDs
        "imdb_id":              imdb_id,
        "tmdb_id":              _safe_int(row.get("tmdbId")) or tmdb_id,
    }
