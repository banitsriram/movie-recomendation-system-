"""
============================================================
  🎬 MOVIE ML RECOMMENDER SYSTEM
  Uses: Pandas, Matplotlib, Scikit-learn
  Dataset: TMDB 5000 Movies (Kaggle)
============================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import ast

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "tmdb_5000_movies.csv")  # CSV lives next to this script
PLOT_STYLE   = "dark_background"
TOP_N        = 10                        # Number of recommendations to show

plt.style.use(PLOT_STYLE)
COLORS = ["#E50914", "#F5C518", "#00B4D8", "#90E0EF", "#FF6B6B",
          "#4ECDC4", "#FFE66D", "#A8DADC", "#FF9F1C", "#CBFF8C"]


# ─────────────────────────────────────────────
#  1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    print("\n📂  Loading dataset ...")
    df = pd.read_csv(path)

    # Parse JSON-like columns
    def extract_names(cell, key="name"):
        try:
            items = ast.literal_eval(cell)
            return [item[key] for item in items] if isinstance(items, list) else []
        except Exception:
            return []

    df["genres_list"]    = df["genres"].apply(extract_names)
    df["keywords_list"]  = df["keywords"].apply(extract_names)

    # Clean / fill
    df["overview"]       = df["overview"].fillna("")
    df["vote_average"] = df["vote_average"].astype(str).apply(pd.to_numeric, errors="coerce").fillna(0)
    df["vote_count"]   = df["vote_count"].astype(str).apply(pd.to_numeric, errors="coerce").fillna(0)
    df["popularity"]   = df["popularity"].astype(str).apply(pd.to_numeric, errors="coerce").fillna(0)
    df["revenue"]      = df["revenue"].astype(str).apply(pd.to_numeric, errors="coerce").fillna(0)
    df["budget"]       = df["budget"].astype(str).apply(pd.to_numeric, errors="coerce").fillna(0)
    df["release_date"]   = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"]           = df["release_date"].dt.year

    # Weighted rating (Bayesian)
    C  = df["vote_average"].mean()
    m  = df["vote_count"].quantile(0.70)
    q  = df[df["vote_count"] >= m].copy()
    q["score"] = (q["vote_count"] / (q["vote_count"] + m)) * q["vote_average"] + \
                 (m / (q["vote_count"] + m)) * C
    df = df.merge(q[["id", "score"]], on="id", how="left")
    df["score"] = df["score"].fillna(0)

    print(f"✅  Loaded {len(df):,} movies.\n")
    return df


# ─────────────────────────────────────────────
#  2. GENRE CLUSTERING (KMeans)
# ─────────────────────────────────────────────
def cluster_movies(df: pd.DataFrame):
    print("🤖  Clustering movies by genre + overview ...")

    # Bag-of-words genre string
    df["genre_str"] = df["genres_list"].apply(lambda g: " ".join(g).lower())
    df["text_soup"] = df["genre_str"] + " " + df["overview"].str.lower()

    tfidf = TfidfVectorizer(max_features=500, stop_words="english")
    matrix = tfidf.fit_transform(df["text_soup"]).toarray()

    # PCA → up to 50 dims, bounded by #samples and #features so small
    # catalogues don't blow up, then KMeans
    n_components = min(50, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(matrix)

    km = KMeans(n_clusters=min(8, len(df)), random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(reduced)
    # Keep the first 2 PCA dims so the scatter plot reflects the real cluster space
    df["pca_x"], df["pca_y"] = reduced[:, 0], reduced[:, 1]

    print("✅  Clustering complete.\n")
    return df, matrix


# ─────────────────────────────────────────────
#  3. CONTENT-BASED RECOMMENDER
# ─────────────────────────────────────────────
def build_recommender(df: pd.DataFrame):
    print("🔗  Building similarity matrix ...")
    df["soup"] = df["genre_str"] + " " + \
                 df["keywords_list"].apply(lambda k: " ".join(k[:10]).lower()) + \
                 " " + df["overview"].str.lower()

    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    matrix = tfidf.fit_transform(df["soup"])
    sim    = cosine_similarity(matrix, matrix)

    df = df.reset_index(drop=True)
    indices = pd.Series(df.index, index=df["title"].str.lower())

    print("✅  Similarity matrix ready.\n")
    return sim, indices


def recommend_by_title(title: str, df, sim, indices, n=TOP_N):
    key = title.strip().lower()
    if key not in indices:
        # fuzzy fallback
        matches = [t for t in indices.index if key in t]
        if not matches:
            print(f"❌  '{title}' not found. Try another title.")
            return pd.DataFrame()
        key = matches[0]
        print(f"🔍  Closest match found: '{df.loc[indices[key], 'title']}'")

    idx   = indices[key]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    scores = list(enumerate(sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in scores]

    cols = ["title", "genres_list", "vote_average", "score", "year"]
    result = df.loc[movie_indices, cols].copy()
    result["similarity"] = [round(s[1], 3) for s in scores]
    return result


def recommend_by_genre(genre: str, df, n=TOP_N):
    genre = genre.strip().lower()
    mask  = df["genres_list"].apply(lambda g: any(genre in x.lower() for x in g))
    subset = df[mask].sort_values("score", ascending=False).head(n)
    if subset.empty:
        print(f"❌  No movies found for genre '{genre}'.")
    return subset[["title", "genres_list", "vote_average", "score", "year"]]


# ─────────────────────────────────────────────
#  4. GRAPHS
# ─────────────────────────────────────────────
def plot_all(df: pd.DataFrame):
    print("📊  Generating visualisations ...")

    # --- 4a. Genre distribution bar chart ---
    from collections import Counter
    all_genres = [g for sublist in df["genres_list"] for g in sublist]
    genre_counts = Counter(all_genres).most_common(15)
    genres, counts = zip(*genre_counts)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(genres, counts, color=COLORS * 2)
    ax.set_xlabel("Number of Movies", color="white")
    ax.set_title("🎭  Top 15 Movie Genres", color="white", fontsize=16, pad=15)
    ax.tick_params(colors="white")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                str(count), va="center", color="white", fontsize=9)
    plt.tight_layout()
    plt.savefig("01_genre_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()

    # --- 4b. Rating distribution ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["vote_average"][df["vote_average"] > 0], bins=30,
            color=COLORS[0], edgecolor="black", alpha=0.85)
    ax.axvline(df["vote_average"].mean(), color=COLORS[1],
               linewidth=2, linestyle="--", label=f"Mean: {df['vote_average'].mean():.2f}")
    ax.set_xlabel("Rating (0–10)", color="white")
    ax.set_ylabel("Count",         color="white")
    ax.set_title("⭐  Vote Average Distribution", color="white", fontsize=16)
    ax.legend(facecolor="#333", labelcolor="white")
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig("02_rating_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()

    # --- 4c. Movies per year ---
    year_counts = df["year"].dropna().astype(int).value_counts().sort_index()
    year_counts = year_counts[year_counts.index >= 1960]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(year_counts.index, year_counts.values, color=COLORS[2], alpha=0.5)
    ax.plot(year_counts.index, year_counts.values, color=COLORS[2], linewidth=1.5)
    ax.set_xlabel("Year",   color="white")
    ax.set_ylabel("Movies", color="white")
    ax.set_title("🗓️  Movies Released Per Year", color="white", fontsize=16)
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig("03_movies_per_year.png", dpi=150, bbox_inches="tight")
    plt.show()

    # --- 4d. Top 10 highest-rated movies (weighted score) ---
    top = df.loc[df["vote_count"] > 500].nlargest(10, "score")
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(top["title"], top["score"], color=COLORS)
    ax.set_xlabel("Weighted Score", color="white")
    ax.set_title("🏆  Top 10 Highest-Rated Movies", color="white", fontsize=16)
    ax.tick_params(colors="white")
    ax.invert_yaxis()
    for bar, val in zip(bars, top["score"]):
        ax.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", ha="right", color="black",
                fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig("04_top_rated.png", dpi=150, bbox_inches="tight")
    plt.show()

    # --- 4e. Cluster scatter (first 2 PCA dims from the actual clustering) ---
    df2 = df.dropna(subset=["cluster"]).copy()

    fig, ax = plt.subplots(figsize=(12, 8))
    for cid in sorted(df2["cluster"].unique()):
        mask = df2["cluster"] == cid
        ax.scatter(df2.loc[mask, "pca_x"], df2.loc[mask, "pca_y"],
                   color=COLORS[cid % len(COLORS)],
                   label=f"Cluster {cid}", alpha=0.5, s=8)
    ax.set_title("🗺️  Movie Clusters (PCA 2D)", color="white", fontsize=16)
    ax.legend(facecolor="#222", labelcolor="white", markerscale=3)
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig("05_clusters.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("✅  All graphs saved as PNG files.\n")


# ─────────────────────────────────────────────
#  5. INTERACTIVE CLI MENU
# ─────────────────────────────────────────────
def print_header():
    print("""
╔══════════════════════════════════════════════════════╗
║        🎬  MOVIE ML RECOMMENDER SYSTEM  🎬           ║
╠══════════════════════════════════════════════════════╣
║  [1]  Get recommendations by MOVIE TITLE             ║
║  [2]  Browse top movies by GENRE                     ║
║  [3]  View DATA VISUALISATIONS                       ║
║  [4]  Search & RATE a movie                          ║
║  [5]  Exit                                           ║
╚══════════════════════════════════════════════════════╝
""")


def display_table(result: pd.DataFrame, title: str = "Results"):
    if result.empty:
        return
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")
    print(f"  {'#':<4} {'Title':<42} {'Year':<7} {'Rating':<8} {'Score'}")
    print(f"  {'─'*65}")
    for i, (_, row) in enumerate(result.iterrows(), 1):
        year  = int(row["year"])  if pd.notna(row.get("year"))  else "N/A"
        score = f"{row['score']:.2f}" if row.get("score", 0) > 0 else "—"
        print(f"  {i:<4} {row['title'][:40]:<42} {str(year):<7} "
              f"{row['vote_average']:<8.1f} {score}")
    print(f"{'═'*70}\n")


def rate_movie(df: pd.DataFrame):
    title = input("Enter movie title to look up: ").strip()
    matches = df[df["title"].str.lower().str.contains(title.lower(), na=False)]
    if matches.empty:
        print("Movie not found.")
        return
    movie = matches.iloc[0]
    print(f"\n🎬  {movie['title']}  ({int(movie['year']) if pd.notna(movie['year']) else 'N/A'})")
    print(f"   Genres   : {', '.join(movie['genres_list'])}")
    print(f"   Overview : {movie['overview'][:200]}...")
    print(f"   Avg Rating (community) : {movie['vote_average']}/10  "
          f"({int(movie['vote_count']):,} votes)")
    print(f"   Weighted Score         : {movie['score']:.2f}")

    your_rating = input("\n⭐  Your rating (1-10, or Enter to skip): ").strip()
    if your_rating.isdigit() and 1 <= int(your_rating) <= 10:
        print(f"\n   Thanks! You rated '{movie['title']}' → {your_rating}/10  🎉")
    else:
        print("   Rating skipped.")


def main():
    # Load
    try:
        df = load_data(DATASET_PATH)
    except FileNotFoundError:
        print(f"\n❌  File '{DATASET_PATH}' not found.")
        print("    Please download 'tmdb_5000_movies.csv' from Kaggle and place it")
        print("    in the same folder as this script.\n")
        return

    df, _ = cluster_movies(df)
    sim, indices = build_recommender(df)

    all_genres = sorted(set(g for sublist in df["genres_list"] for g in sublist))

    while True:
        print_header()
        choice = input("Select option (1-5): ").strip()

        if choice == "1":
            title = input("Enter a movie title: ").strip()
            recs  = recommend_by_title(title, df, sim, indices)
            display_table(recs, f"Movies similar to '{title}'")

        elif choice == "2":
            print(f"\n  Available genres: {', '.join(all_genres)}\n")
            genre = input("Enter genre: ").strip()
            recs  = recommend_by_genre(genre, df)
            display_table(recs, f"Top '{genre}' movies")

        elif choice == "3":
            plot_all(df)

        elif choice == "4":
            rate_movie(df)

        elif choice == "5":
            print("\n👋  Goodbye! Enjoy your movies.\n")
            break

        else:
            print("⚠️  Invalid option. Please enter 1-5.")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()