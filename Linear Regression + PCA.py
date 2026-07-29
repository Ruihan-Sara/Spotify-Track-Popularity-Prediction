import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================
# 1. Load data
# =========================
df = pd.read_csv("songs_merged.csv")

print("Original shape:", df.shape)

# =========================
# 2. Basic cleaning
# =========================

# Drop unnamed column if it exists
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Create duration in minutes if duration_ms exists
if "duration_ms" in df.columns:
    df["duration_min"] = df["duration_ms"] / 60000

# Convert explicit to int if it exists
if "explicit" in df.columns:
    try:
        df["explicit"] = df["explicit"].astype(int)
    except Exception:
        pass

# Log-transform market variables if they exist
if "followers" in df.columns:
    df["log_followers"] = np.log(df["followers"] + 1)

if "monthly_listeners" in df.columns:
    df["log_monthly_listeners"] = np.log(df["monthly_listeners"] + 1)

# =========================
# 3. Define feature sets
# =========================

audio_features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

market_features = []

if "popularity_artist" in df.columns:
    market_features.append("popularity_artist")

if "log_followers" in df.columns:
    market_features.append("log_followers")

if "log_monthly_listeners" in df.columns:
    market_features.append("log_monthly_listeners")

all_features = audio_features + market_features

print("\nAudio features used:", audio_features)
print("Market features used:", market_features)

# =========================
# 4. Keep only required columns
# =========================

required_cols = ["popularity"] + all_features
df_model = df[required_cols].copy()

# Drop rows with missing values in required columns
df_model = df_model.dropna()

print("\nModeling shape after dropping missing values:", df_model.shape)

# =========================
# 5. Define X and y
# =========================

X_audio = df_model[audio_features]
X_all = df_model[all_features]
y = df_model["popularity"]

# =========================
# 6. Train-test split
# =========================

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X_audio, y, test_size=0.2, random_state=42
)

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all, y, test_size=0.2, random_state=42
)

# =========================
# 7. Standardization
# =========================

scaler_a = StandardScaler()
X_train_a_scaled = scaler_a.fit_transform(X_train_a)
X_test_a_scaled = scaler_a.transform(X_test_a)

scaler_all = StandardScaler()
X_train_all_scaled = scaler_all.fit_transform(X_train_all)
X_test_all_scaled = scaler_all.transform(X_test_all)

# =========================
# 8. PCA
# =========================

n_audio_components = min(5, X_train_a_scaled.shape[1])
pca_a = PCA(n_components=n_audio_components)

X_train_a_pca = pca_a.fit_transform(X_train_a_scaled)
X_test_a_pca = pca_a.transform(X_test_a_scaled)

n_all_components = min(5, X_train_all_scaled.shape[1])
pca_all = PCA(n_components=n_all_components)

X_train_all_pca = pca_all.fit_transform(X_train_all_scaled)
X_test_all_pca = pca_all.transform(X_test_all_scaled)

# =========================
# 9. Regression
# =========================

model_audio = LinearRegression()
model_audio.fit(X_train_a_pca, y_train_a)
y_pred_audio = model_audio.predict(X_test_a_pca)

model_all = LinearRegression()
model_all.fit(X_train_all_pca, y_train_all)
y_pred_all = model_all.predict(X_test_all_pca)

# =========================
# 10. Evaluation
# =========================

mse_audio = mean_squared_error(y_test_a, y_pred_audio)
mae_audio = mean_absolute_error(y_test_a, y_pred_audio)
r2_audio = r2_score(y_test_a, y_pred_audio)

mse_all = mean_squared_error(y_test_all, y_pred_all)
mae_all = mean_absolute_error(y_test_all, y_pred_all)
r2_all = r2_score(y_test_all, y_pred_all)

print("\nFinal Model Results")
print(f"Audio-only Model -> MSE: {mse_audio:.3f}, MAE: {mae_audio:.3f}, R²: {r2_audio:.4f}")
print(f"Audio + Market Model -> MSE: {mse_all:.3f}, MAE: {mae_all:.3f}, R²: {r2_all:.4f}")

# =========================
# 11. PCA explained variance
# =========================

print("\nExplained variance ratio (Audio-only PCA):")
print(pca_a.explained_variance_ratio_)

print("\nExplained variance ratio (Audio + Market PCA):")
print(pca_all.explained_variance_ratio_)