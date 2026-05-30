"""
Hermetic pipeline tests — run anywhere, no dataset download required.

These use a tiny synthetic dataset (tests/sample_movies.csv) that mirrors
the TMDB 5000 schema, so the full load → cluster → recommend pipeline is
exercised on every `git clone` and in CI. They check mechanics (shapes,
parsing, edge cases) rather than real-world relevance, which is covered by
the integration tests in test_recommender.py.
"""

import os

import pytest

import main

SAMPLE = os.path.join(os.path.dirname(__file__), "sample_movies.csv")


@pytest.fixture(scope="module")
def model():
    df = main.load_data(SAMPLE)
    df, _ = main.cluster_movies(df)
    sim, indices = main.build_recommender(df)
    return df, sim, indices


def test_load_parses_schema(model):
    df, _, _ = model
    assert len(df) == 12
    for col in ("genres_list", "keywords_list", "score", "year", "soup"):
        assert col in df.columns
    # JSON-like columns are parsed into real lists
    assert isinstance(df["genres_list"].iloc[0], list)
    assert df["genres_list"].iloc[0]  # non-empty


def test_recommend_returns_requested_count(model):
    df, sim, indices = model
    recs = main.recommend_by_title("Sample Movie 1", df, sim, indices, n=3)
    assert len(recs) == 3
    # never recommends the query movie itself
    assert "Sample Movie 1" not in list(recs["title"])


def test_recommend_unknown_title_is_empty(model):
    df, sim, indices = model
    recs = main.recommend_by_title("definitely not in the data", df, sim, indices, n=3)
    assert recs.empty


def test_genre_filter_matches(model):
    df, _, _ = model
    recs = main.recommend_by_genre("Action", df, n=10)
    assert not recs.empty
    assert all("Action" in g for g in recs["genres_list"])


def test_unknown_genre_is_empty(model):
    df, _, _ = model
    assert main.recommend_by_genre("nosuchgenre", df).empty


def test_similarity_self_is_highest(model):
    _, sim, _ = model
    assert sim[0].argmax() == 0
