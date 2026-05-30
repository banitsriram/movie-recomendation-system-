"""
Integration tests against the real TMDB 5000 dataset.

These assert on real-world results (e.g. a "Batman" query returns Batman
films), so they need the actual dataset. If it isn't present (e.g. on a
fresh clone or in CI) the whole module is skipped — see test_pipeline.py
for hermetic tests that always run.

The model is built once per module since loading + vectorising the full
dataset takes a few seconds.
"""

import os

import pytest

import main

pytestmark = pytest.mark.skipif(
    not os.path.exists(main.DATASET_PATH),
    reason="TMDB 5000 dataset not found — download it to run integration tests (see README).",
)


@pytest.fixture(scope="module")
def model():
    df = main.load_data(main.DATASET_PATH)
    df, _ = main.cluster_movies(df)
    sim, indices = main.build_recommender(df)
    return df, sim, indices


def test_data_loads(model):
    df, _, _ = model
    assert len(df) > 4000
    # required derived columns exist
    for col in ("genres_list", "score", "year", "soup"):
        assert col in df.columns


def test_recommend_exact_title(model):
    df, sim, indices = model
    recs = main.recommend_by_title("The Dark Knight", df, sim, indices, n=5)
    assert len(recs) == 5
    # should not recommend the movie itself
    assert "The Dark Knight" not in list(recs["title"])
    # batman query should surface at least one other batman film
    assert any("batman" in t.lower() for t in recs["title"])


def test_recommend_fuzzy_fallback(model):
    df, sim, indices = model
    # lower-case partial title should still resolve via fuzzy fallback
    recs = main.recommend_by_title("avatar", df, sim, indices, n=3)
    assert not recs.empty


def test_recommend_not_found(model):
    df, sim, indices = model
    recs = main.recommend_by_title("zzzz not a real movie", df, sim, indices, n=3)
    assert recs.empty


def test_recommend_by_genre(model):
    df, _, _ = model
    recs = main.recommend_by_genre("Action", df, n=5)
    assert len(recs) == 5
    # every result should actually be an action movie
    assert all("Action" in g for g in recs["genres_list"])


def test_genre_not_found(model):
    df, _, _ = model
    recs = main.recommend_by_genre("notarealgenre", df)
    assert recs.empty


def test_similarity_self_is_highest(model):
    df, sim, _ = model
    # a movie is most similar to itself
    assert sim[0].argmax() == 0
