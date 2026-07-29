import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Load data
# =========================
df = pd.read_csv("songs_merged.csv")

print("First 5 rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nInfo:")
df.info()

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False))

print("\nDuplicate rows:")
print(df.duplicated().sum())

# =========================
# 2. Basic cleaning for EDA
# =========================

# Drop unnamed index column if it exists
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Convert release_date to datetime if it exists
if "release_date" in df.columns:
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year

# Convert duration to minutes if duration_ms exists
if "duration_ms" in df.columns:
    df["duration_min"] = df["duration_ms"] / 60000

# Convert explicit to int if it exists
if "explicit" in df.columns:
    try:
        df["explicit"] = df["explicit"].astype(int)
    except Exception:
        pass

# Create log-transformed market variables if they exist
if "followers" in df.columns:
    df["log_followers"] = np.log(df["followers"] + 1)

if "monthly_listeners" in df.columns:
    df["log_monthly_listeners"] = np.log(df["monthly_listeners"] + 1)

# Use a plotting dataframe to remove extreme duration outliers for clearer charts
df_plot = df.copy()
if "duration_min" in df_plot.columns:
    df_plot = df_plot[df_plot["duration_min"] < 10]

# =========================
# 3. Summary statistics
# =========================
summary_candidates = [
    "popularity",
    "duration_min",
    "danceability",
    "energy",
    "acousticness",
    "speechiness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "followers",
    "monthly_listeners",
    "popularity_artist",
    "log_followers",
    "log_monthly_listeners",
]

summary_cols = [col for col in summary_candidates if col in df.columns]

print("\nSummary statistics:")
print(df[summary_cols].describe())

# =========================
# 4. Histograms
# =========================
hist_candidates = [
    ("popularity", "Distribution of Track Popularity"),
    ("duration_min", "Distribution of Song Duration (minutes)"),
    ("danceability", "Distribution of Danceability"),
    ("energy", "Distribution of Energy"),
    ("log_followers", "Distribution of Log Followers"),
    ("log_monthly_listeners", "Distribution of Log Monthly Listeners"),
    ("popularity_artist", "Distribution of Artist Popularity"),
]

for col, title in hist_candidates:
    if col in df_plot.columns:
        plt.figure(figsize=(6, 4))
        plt.hist(df_plot[col].dropna(), bins=30)
        plt.title(title)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

# =========================
# 5. Correlation heatmap
# =========================
corr_candidates = [
    "popularity",
    "duration_min",
    "danceability",
    "energy",
    "acousticness",
    "speechiness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "monthly_listeners",
    "popularity_artist",
    "followers",
    "release_year",
    "explicit",
]

corr_cols = [col for col in corr_candidates if col in df.columns]

if len(corr_cols) > 1:
    corr = df[corr_cols].corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Matrix of Audio and Market Features")
    plt.tight_layout()
    plt.show()

# =========================
# 6. Scatter plots: Audio vs Popularity
# =========================
scatter_audio = [
    ("energy", "Energy vs Popularity"),
    ("danceability", "Danceability vs Popularity"),
    ("valence", "Valence vs Popularity"),
]

for xcol, title in scatter_audio:
    if xcol in df.columns and "popularity" in df.columns:
        plt.figure(figsize=(6, 4))
        plt.scatter(df[xcol], df["popularity"], alpha=0.2)
        plt.xlabel(xcol)
        plt.ylabel("Popularity")
        plt.title(title)
        plt.tight_layout()
        plt.show()

# =========================
# 7. Scatter plots: Market vs Popularity
# =========================
scatter_market = [
    ("log_followers", "Log Followers vs Popularity"),
    ("log_monthly_listeners", "Log Monthly Listeners vs Popularity"),
    ("popularity_artist", "Artist Popularity vs Track Popularity"),
]

for xcol, title in scatter_market:
    if xcol in df.columns and "popularity" in df.columns:
        plt.figure(figsize=(6, 4))
        plt.scatter(df[xcol], df["popularity"], alpha=0.2)
        plt.xlabel(xcol)
        plt.ylabel("Popularity")
        plt.title(title)
        plt.tight_layout()
        plt.show()

# =========================
# 8. Explicit vs Non-Explicit
# =========================
if "explicit" in df.columns and "popularity" in df.columns:
    explicit_popularity = df.groupby("explicit")["popularity"].mean()

    plt.figure(figsize=(6, 4))
    plt.bar(explicit_popularity.index.astype(str), explicit_popularity.values)
    plt.title("Average Popularity: Explicit vs Non-Explicit")
    plt.xlabel("Explicit")
    plt.ylabel("Average Popularity")
    plt.tight_layout()
    plt.show()

# =========================
# 9. Top genres by average popularity
# =========================
if "genres" in df.columns and "popularity" in df.columns:
    genre_popularity = (
        df.dropna(subset=["genres"])
        .groupby("genres")["popularity"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 5))
    plt.bar(genre_popularity.index, genre_popularity.values)
    plt.title("Top 10 Genres by Average Track Popularity")
    plt.xlabel("Genre")
    plt.ylabel("Average Popularity")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# =========================
# 10. Top artists by average popularity
# =========================
if "names_artist" in df.columns and "popularity" in df.columns:
    artist_popularity = (
        df.groupby("names_artist")["popularity"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 5))
    plt.bar(artist_popularity.index, artist_popularity.values)
    plt.title("Top 10 Artists by Average Track Popularity")
    plt.xlabel("Artist")
    plt.ylabel("Average Popularity")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# =========================
# 11. Playlist effect
# =========================
if "playlists_found" in df.columns and "popularity" in df.columns:
    playlist_popularity = (
        df.groupby("playlists_found")["popularity"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 5))
    plt.bar(playlist_popularity.index.astype(str), playlist_popularity.values)
    plt.title("Average Popularity by Playlist Category")
    plt.xlabel("Playlist Category")
    plt.ylabel("Average Popularity")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

print("\nEDA complete.")