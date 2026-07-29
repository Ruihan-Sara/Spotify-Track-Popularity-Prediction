# ============================================================
#  Spotify Popularity Prediction — Gradient Boosting Model v2
#  Full evaluation + analysis version
# ============================================================

# ── 0. IMPORTS ───────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr


# ── 1. LOAD DATA ─────────────────────────────────────────────
df = pd.read_csv('/Users/zdexx/Desktop/TCD/STU33005/Assignment/songs_merged.csv')
print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")


# ── 2. CLEANING ──────────────────────────────────────────────
df = df.drop_duplicates(subset='ids').reset_index(drop=True)
required = ['popularity', 'monthly_listeners', 'followers', 'popularity_artist']
df = df.dropna(subset=required).reset_index(drop=True)

df['log_followers']         = np.log1p(df['followers'])
df['log_monthly_listeners'] = np.log1p(df['monthly_listeners'])

audio_cols = [
    'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
    'musicalkey', 'musicalmode', 'time_signature', 'duration_ms'
]
df[audio_cols] = df[audio_cols].fillna(df[audio_cols].mean())
print(f"After cleaning: {df.shape[0]:,} rows remain\n")


# ── 3. FEATURE DEFINITIONS ───────────────────────────────────
audio_features = [
    'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
    'musicalkey', 'musicalmode', 'time_signature', 'duration_ms'
]
artist_features = [
    'log_followers', 'log_monthly_listeners',
    'popularity_artist', 'num_releases', 'num_tracks'
]
combined_features = audio_features + artist_features
target = 'popularity'

X_audio    = df[audio_features]
X_combined = df[combined_features]
y          = df[target]


# ── 4. TRAIN / TEST SPLIT ────────────────────────────────────
X_a_train, X_a_test, y_train, y_test = train_test_split(
    X_audio,    y, test_size=0.2, random_state=42)
X_c_train, X_c_test, _,      _       = train_test_split(
    X_combined, y, test_size=0.2, random_state=42)


# ── 5. MODEL TRAINING ────────────────────────────────────────
hgb_params = dict(
    max_iter=500, max_depth=4, learning_rate=0.05,
    l2_regularization=0.1, min_samples_leaf=20,
    early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=25, random_state=42
)

print("Training Model 1 (Audio only)...")
model_audio = HistGradientBoostingRegressor(**hgb_params)
model_audio.fit(X_a_train, y_train)
print(f"  → stopped at iteration {model_audio.n_iter_}")

print("Training Model 2 (Audio + Artist)...")
model_combined = HistGradientBoostingRegressor(**hgb_params)
model_combined.fit(X_c_train, y_train)
print(f"  → stopped at iteration {model_combined.n_iter_}\n")


# ── 6. EVALUATION ────────────────────────────────────────────
def evaluate(model, X_train, X_test, y_train, y_test, label):
    preds = model.predict(X_test)
    r2    = r2_score(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = np.mean(np.abs(y_test - preds))
    bias  = np.mean(preds - y_test)          # systematic over/under prediction

    kf    = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2', n_jobs=-1)
    cv_rmse = cross_val_score(model, X_train, y_train, cv=kf,
                              scoring='neg_root_mean_squared_error', n_jobs=-1)

    pearson_r,  p_pearson  = pearsonr(y_test, preds)
    spearman_r, p_spearman = spearmanr(y_test, preds)

    print(f"{'─'*50}")
    print(f"  {label}")
    print(f"  Test R²              : {r2:.4f}")
    print(f"  Test RMSE            : {rmse:.4f}")
    print(f"  Test MAE             : {mae:.4f}")
    print(f"  Prediction Bias      : {bias:+.4f}  (+ = overpredict)")
    print(f"  CV R²  (5-fold)      : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    print(f"  CV RMSE (5-fold)     : {(-cv_rmse).mean():.4f} ± {cv_rmse.std():.4f}")
    print(f"  Pearson r            : {pearson_r:.4f}  (p={p_pearson:.2e})")
    print(f"  Spearman ρ           : {spearman_r:.4f}  (p={p_spearman:.2e})")
    return preds, r2, rmse

preds_audio,    r2_a, rmse_a = evaluate(model_audio,    X_a_train, X_a_test, y_train, y_test, "Model 1 — Audio Only")
preds_combined, r2_c, rmse_c = evaluate(model_combined, X_c_train, X_c_test, y_train, y_test, "Model 2 — Audio + Artist")

print(f"\n  ▶ R² lift from artist features : +{r2_c - r2_a:.4f}")
print(f"  ▶ RMSE reduction               : -{rmse_a - rmse_c:.4f} points\n")


# ── 7. PERMUTATION IMPORTANCE (works for HistGBR) ────────────
print("Computing permutation importances (this takes ~30s)...")

pi_audio = permutation_importance(
    model_audio, X_a_test, y_test,
    n_repeats=10, random_state=42, n_jobs=-1)

pi_combined = permutation_importance(
    model_combined, X_c_test, y_test,
    n_repeats=10, random_state=42, n_jobs=-1)

imp_audio = pd.Series(pi_audio.importances_mean,    index=audio_features).sort_values()
imp_comb  = pd.Series(pi_combined.importances_mean, index=combined_features).sort_values()


# ── 8. FAME PREMIUM ──────────────────────────────────────────
df['pred_audio']   = model_audio.predict(X_audio)
df['fame_premium'] = df['popularity'] - df['pred_audio']

print("── Top 10 market-driven songs (Fame Premium ↑) ──")
cols = ['names', 'names_artist', 'popularity', 'pred_audio', 'fame_premium']
print(df[cols].sort_values('fame_premium', ascending=False).head(10).to_string(index=False))

print("\n── Top 10 undervalued songs (audio > fame, popularity > 0) ──")
# Exclude popularity=0 (likely unlisted/removed tracks, not truly undervalued)
undervalued = df[df['popularity'] > 0].sort_values('fame_premium').head(10)
print(undervalued[cols].to_string(index=False))

# Fame premium summary stats
print(f"\n── Fame Premium Statistics ──")
print(f"  Mean   : {df['fame_premium'].mean():+.2f}")
print(f"  Median : {df['fame_premium'].median():+.2f}")
print(f"  Std    : {df['fame_premium'].std():.2f}")
print(f"  % songs with positive premium (market > audio): "
      f"{(df['fame_premium'] > 0).mean()*100:.1f}%")


# ── 9. VISUALISATIONS ────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
fig = plt.figure(figsize=(20, 18))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── 9a. Feature Importance — Model 1 ─────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
imp_audio.plot(kind='barh', ax=ax1, color='steelblue', xerr=pd.Series(
    pi_audio.importances_std, index=audio_features).reindex(imp_audio.index))
ax1.set_title('Permutation Importance\nModel 1: Audio Only', fontweight='bold')
ax1.set_xlabel('Mean decrease in R² when feature is shuffled')
ax1.axvline(0, color='black', linewidth=0.8, linestyle='--')

# ── 9b. Feature Importance — Model 2 ─────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
colors = ['coral' if f in artist_features else 'steelblue' for f in imp_comb.index]
imp_comb.plot(kind='barh', ax=ax2, color=colors, xerr=pd.Series(
    pi_combined.importances_std, index=combined_features).reindex(imp_comb.index))
ax2.set_title('Permutation Importance\nModel 2: Audio + Artist  [coral = artist]',
              fontweight='bold')
ax2.set_xlabel('Mean decrease in R² when feature is shuffled')
ax2.axvline(0, color='black', linewidth=0.8, linestyle='--')

# ── 9c. Actual vs Predicted — Model 1 ────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(y_test, preds_audio, alpha=0.25, s=10, color='steelblue')
lims = [0, 100]
ax3.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
ax3.set_xlabel('Actual Popularity')
ax3.set_ylabel('Predicted Popularity')
ax3.set_title(f'Actual vs Predicted — Model 1\n(Test R² = {r2_a:.3f})', fontweight='bold')
ax3.set_xlim(lims); ax3.set_ylim(lims)
ax3.legend()

# ── 9d. Actual vs Predicted — Model 2 ────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(y_test, preds_combined, alpha=0.25, s=10, color='teal')
ax4.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
ax4.set_xlabel('Actual Popularity')
ax4.set_ylabel('Predicted Popularity')
ax4.set_title(f'Actual vs Predicted — Model 2\n(Test R² = {r2_c:.3f})', fontweight='bold')
ax4.set_xlim(lims); ax4.set_ylim(lims)
ax4.legend()

# ── 9e. Fame Premium Distribution ────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
sns.histplot(df['fame_premium'], bins=60, kde=True, ax=ax5, color='mediumpurple')
ax5.axvline(0,  color='red',   linestyle='--', linewidth=1.5, label='Zero premium')
ax5.axvline(df['fame_premium'].median(), color='orange',
            linestyle=':', linewidth=1.5, label=f"Median ({df['fame_premium'].median():.1f})")
ax5.set_title('Fame Premium Distribution\n(Actual − Audio-predicted Popularity)',
              fontweight='bold')
ax5.set_xlabel('Fame Premium')
ax5.legend()

# ── 9f. Fame Premium vs Artist Popularity scatter ────────────
ax6 = fig.add_subplot(gs[2, 1])
sc = ax6.scatter(df['popularity_artist'], df['fame_premium'],
                 alpha=0.2, s=8, c=df['popularity'], cmap='RdYlGn',
                 vmin=0, vmax=100)
plt.colorbar(sc, ax=ax6, label='Track Popularity')
ax6.axhline(0, color='red', linestyle='--', linewidth=1)
ax6.set_xlabel('Artist Popularity Score')
ax6.set_ylabel('Fame Premium')
ax6.set_title('Fame Premium vs Artist Popularity\n(color = track popularity)',
              fontweight='bold')

plt.suptitle('Spotify Popularity — Gradient Boosting Full Analysis',
             fontsize=15, fontweight='bold', y=1.01)
plt.savefig('gbm_results_v2.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✓ Plot saved to gbm_results_v2.png")


