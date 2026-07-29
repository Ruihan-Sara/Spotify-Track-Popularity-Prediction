# ============================================================
#  Spotify GBM — Extended Evaluation & Analysis
#  Residual Analysis | Confusion Matrix | SHAP | Error Analysis | Distribution
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_squared_error,
                              classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from scipy import stats
import shap

# ── OUTPUT FOLDER (saves all plots here) ─────────────────────
import os
OUT = '/Users/zdexx/Desktop/TCD/STU33005/Assignment/analysis_plots/'
os.makedirs(OUT, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  SECTION 1 — DATA & MODEL  (same as main script)
# ══════════════════════════════════════════════════════════════
print("="*55)
print("  SECTION 1 — Loading data & training models")
print("="*55)

df = pd.read_csv('/Users/zdexx/Desktop/TCD/STU33005/Assignment/songs_merged.csv')
df = df.drop_duplicates(subset='ids').reset_index(drop=True)
df = df.dropna(subset=['popularity','monthly_listeners','followers','popularity_artist']).reset_index(drop=True)

df['log_followers']         = np.log1p(df['followers'])
df['log_monthly_listeners'] = np.log1p(df['monthly_listeners'])

audio_cols = ['acousticness','danceability','energy','instrumentalness',
              'liveness','loudness','speechiness','tempo','valence',
              'musicalkey','musicalmode','time_signature','duration_ms']
df[audio_cols] = df[audio_cols].fillna(df[audio_cols].mean())
print(f"  Dataset: {df.shape[0]:,} rows after cleaning")

audio_features   = audio_cols
artist_features  = ['log_followers','log_monthly_listeners',
                    'popularity_artist','num_releases','num_tracks']
combined_features = audio_features + artist_features

X_audio    = df[audio_features]
X_combined = df[combined_features]
y          = df['popularity']

# Keep track of original df indices through the split
df = df.reset_index(drop=True)
idx = df.index

idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

X_a_train = X_audio.loc[idx_train];    X_a_test = X_audio.loc[idx_test]
X_c_train = X_combined.loc[idx_train]; X_c_test = X_combined.loc[idx_test]
y_train   = y.loc[idx_train];          y_test   = y.loc[idx_test]

hgb_params = dict(
    max_iter=500, max_depth=4, learning_rate=0.05,
    l2_regularization=0.1, min_samples_leaf=20,
    early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=25, random_state=42
)

print("  Training Model 1 (Audio only)...")
model_audio = HistGradientBoostingRegressor(**hgb_params)
model_audio.fit(X_a_train, y_train)

print("  Training Model 2 (Audio + Artist)...")
model_combined = HistGradientBoostingRegressor(**hgb_params)
model_combined.fit(X_c_train, y_train)

preds_audio    = model_audio.predict(X_a_test)
preds_combined = model_combined.predict(X_c_test)
residuals      = y_test.values - preds_combined

print(f"  Model 1 R²: {r2_score(y_test, preds_audio):.4f}")
print(f"  Model 2 R²: {r2_score(y_test, preds_combined):.4f}\n")


# ══════════════════════════════════════════════════════════════
#  SECTION 2 — RESIDUAL ANALYSIS
# ══════════════════════════════════════════════════════════════
print("="*55)
print("  SECTION 2 — Residual Analysis")
print("="*55)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Residual Analysis — Model 2 (Audio + Artist)',
             fontsize=13, fontweight='bold')

# 2a. Residuals vs Predicted
axes[0].scatter(preds_combined, residuals, alpha=0.25, s=8, color='teal')
axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
# Lowess smoothing line to show trend
from statsmodels.nonparametric.smoothers_lowess import lowess
sorted_idx = np.argsort(preds_combined)
lw = lowess(residuals[sorted_idx], preds_combined[sorted_idx], frac=0.3)
axes[0].plot(lw[:,0], lw[:,1], color='orange', linewidth=2, label='Trend (lowess)')
axes[0].set_xlabel('Predicted Popularity')
axes[0].set_ylabel('Residual  (Actual − Predicted)')
axes[0].set_title('Residuals vs Predicted\n(should be random around 0)')
axes[0].legend(fontsize=8)

# 2b. Residual Distribution
sns.histplot(residuals, kde=True, ax=axes[1], color='mediumpurple', bins=50)
axes[1].axvline(0,                    color='red',    linestyle='--', linewidth=1.5, label='Zero')
axes[1].axvline(residuals.mean(),     color='orange', linestyle=':',  linewidth=1.5,
                label=f'Mean ({residuals.mean():.2f})')
axes[1].axvline(np.median(residuals), color='green',  linestyle=':',  linewidth=1.5,
                label=f'Median ({np.median(residuals):.2f})')
axes[1].set_title('Residual Distribution\n(should be ~normal, centred at 0)')
axes[1].set_xlabel('Residual')
axes[1].legend(fontsize=8)

# Normality test
stat, p_norm = stats.shapiro(residuals[:500])   # Shapiro uses ≤5000 samples
print(f"  Shapiro-Wilk normality test (n=500 sample): W={stat:.4f}, p={p_norm:.4e}")
if p_norm < 0.05:
    print("  → Residuals are NOT perfectly normal (common in real-world data)")
else:
    print("  → Residuals are approximately normal")

# 2c. Q-Q Plot
stats.probplot(residuals, plot=axes[2])
axes[2].set_title('Q-Q Plot\n(points on line = normal residuals)')
axes[2].get_lines()[1].set_color('red')

plt.tight_layout()
plt.savefig(OUT + '1_residual_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  ✓ Saved: 1_residual_analysis.png\n")


# ══════════════════════════════════════════════════════════════
#  SECTION 3 — CONFUSION MATRIX (Popularity Tier Classification)
# ══════════════════════════════════════════════════════════════
print("="*55)
print("  SECTION 3 — Confusion Matrix (Tier Classification)")
print("="*55)

def to_tier(x):
    if x < 40:   return 'Low (<40)'
    elif x < 70: return 'Mid (40–70)'
    else:        return 'High (>70)'

y_tier_true   = pd.Series(y_test.values).apply(to_tier)
y_tier_pred_1 = pd.Series(preds_audio).apply(to_tier)
y_tier_pred_2 = pd.Series(preds_combined).apply(to_tier)
tier_order    = ['Low (<40)', 'Mid (40–70)', 'High (>70)']

print("\n  ── Classification Report: Model 1 (Audio Only) ──")
print(classification_report(y_tier_true, y_tier_pred_1, target_names=tier_order))

print("  ── Classification Report: Model 2 (Audio + Artist) ──")
print(classification_report(y_tier_true, y_tier_pred_2, target_names=tier_order))

# Tier distribution in test set
print("  ── Actual Tier Distribution in Test Set ──")
print(y_tier_true.value_counts().to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Confusion Matrix — Popularity Tier Prediction\n'
             '(Low <40 | Mid 40–70 | High >70)',
             fontsize=12, fontweight='bold')

for ax, preds_tier, title in zip(
    axes,
    [y_tier_pred_1, y_tier_pred_2],
    ['Model 1: Audio Only', 'Model 2: Audio + Artist']
):
    cm = confusion_matrix(y_tier_true, preds_tier, labels=tier_order)
    # Normalise rows → shows recall per tier
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=tier_order, yticklabels=tier_order,
                ax=ax, vmin=0, vmax=1,
                annot_kws={'size': 11})
    # Add raw counts in parentheses
    for i in range(3):
        for j in range(3):
            ax.text(j+0.5, i+0.72, f'(n={cm[i,j]})',
                    ha='center', va='center', fontsize=8, color='grey')

    ax.set_xlabel('Predicted Tier')
    ax.set_ylabel('Actual Tier')
    ax.set_title(title, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT + '2_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  ✓ Saved: 2_confusion_matrix.png\n")


# ══════════════════════════════════════════════════════════════
#  SECTION 4 — SHAP VALUES
# ══════════════════════════════════════════════════════════════
print("="*55)
print("  SECTION 4 — SHAP Values (this takes ~1–2 min)")
print("="*55)

# Use a sample for speed (SHAP on full test set can be slow)
shap_sample = X_c_test.sample(n=min(500, len(X_c_test)), random_state=42)

explainer   = shap.TreeExplainer(model_combined)
shap_values = explainer.shap_values(shap_sample)

print(f"  SHAP computed on {len(shap_sample)} test samples")
base_val = float(np.mean(explainer.expected_value))
print(f"  Expected value (base): {base_val:.2f}")

# 4a. Summary plot — beeswarm (global importance + direction)
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, shap_sample,
                  feature_names=combined_features,
                  show=False, plot_size=None)
plt.title('SHAP Summary Plot — Model 2 (Audio + Artist)\n'
          'Each dot = one song; color = feature value; x-position = SHAP impact',
          fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + '3a_shap_summary_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()

# 4b. Bar plot — mean absolute SHAP (clean feature ranking)
plt.figure(figsize=(9, 6))
shap.summary_plot(shap_values, shap_sample,
                  feature_names=combined_features,
                  plot_type='bar', show=False, plot_size=None)
plt.title('SHAP Feature Importance (Mean |SHAP|) — Model 2',
          fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + '3b_shap_importance_bar.png', dpi=150, bbox_inches='tight')
plt.show()

# 4c. Waterfall for best-predicted song and worst-predicted song
errors_sample = np.abs(y_test.loc[shap_sample.index].values -
                       model_combined.predict(shap_sample))

best_local  = np.argmin(errors_sample)
worst_local = np.argmax(errors_sample)

for label, idx_local in [('Best Predicted', best_local), ('Worst Predicted', worst_local)]:
    song_name = df.loc[shap_sample.index[idx_local], 'names']
    artist    = df.loc[shap_sample.index[idx_local], 'names_artist']
    actual    = y_test.loc[shap_sample.index[idx_local]]
    predicted = model_combined.predict(shap_sample.iloc[[idx_local]])[0]

    print(f"\n  {label} song: '{song_name}' by {artist}")
    print(f"  Actual={actual:.1f}  Predicted={predicted:.1f}  Error={actual-predicted:.1f}")

    exp = shap.Explanation(
        values       = shap_values[idx_local],
        base_values  = base_val,
        data         = shap_sample.iloc[idx_local].values,
        feature_names= combined_features
    )
    plt.figure(figsize=(10, 5))
    shap.waterfall_plot(exp, show=False, max_display=12)
    plt.title(f'SHAP Waterfall — {label}\n"{song_name}" by {artist}  '
              f'(Actual={actual:.0f}, Predicted={predicted:.0f})',
              fontsize=10, fontweight='bold')
    plt.tight_layout()
    fname = '3c_shap_waterfall_best.png' if label == 'Best Predicted' else '3d_shap_waterfall_worst.png'
    plt.savefig(OUT + fname, dpi=150, bbox_inches='tight')
    plt.show()

print(f"\n  ✓ Saved: 3a/3b/3c/3d SHAP plots\n")


# ══════════════════════════════════════════════════════════════
#  SECTION 5 — ERROR ANALYSIS BY SUBGROUP
# ══════════════════════════════════════════════════════════════
print("="*55)
print("  SECTION 5 — Error Analysis by Subgroup")
print("="*55)

test_df = df.loc[idx_test].copy()
test_df['pred']      = preds_combined
test_df['residual']  = y_test.values - preds_combined
test_df['abs_error'] = np.abs(test_df['residual'])

# Artist tier
test_df['artist_tier'] = pd.cut(
    test_df['popularity_artist'],
    bins=[0, 40, 70, 100],
    labels=['Low (0–40)', 'Mid (41–70)', 'High (71–100)']
)

# Actual popularity tier
test_df['pop_tier'] = test_df['popularity'].apply(to_tier)

print("\n  ── MAE & Bias by Artist Popularity Tier ──")
tier_summary = test_df.groupby('artist_tier', observed=True).agg(
    Count       = ('abs_error', 'count'),
    MAE         = ('abs_error', 'mean'),
    Bias        = ('residual',  'mean'),
    Std_Error   = ('residual',  'std')
).round(3)
print(tier_summary.to_string())

print("\n  ── MAE & Bias by Actual Popularity Tier ──")
pop_summary = test_df.groupby('pop_tier').agg(
    Count       = ('abs_error', 'count'),
    MAE         = ('abs_error', 'mean'),
    Bias        = ('residual',  'mean'),
    Std_Error   = ('residual',  'std')
).round(3)
print(pop_summary.to_string())

print("\n  ── 20 Worst Predicted Songs ──")
worst20 = test_df.nlargest(20, 'abs_error')[
    ['names','names_artist','popularity','pred','residual','popularity_artist']
].round(1)
print(worst20.to_string(index=False))

# Visualise
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Error Analysis by Subgroup — Model 2', fontsize=13, fontweight='bold')

# 5a. MAE by artist tier
tier_summary['MAE'].plot(kind='bar', ax=axes[0], color=['#4878CF','#6ACC65','#D65F5F'],
                          edgecolor='white', width=0.5)
axes[0].set_title('MAE by Artist Popularity Tier', fontweight='bold')
axes[0].set_ylabel('Mean Absolute Error')
axes[0].set_xlabel('Artist Tier')
axes[0].tick_params(axis='x', rotation=20)
for i, v in enumerate(tier_summary['MAE']):
    axes[0].text(i, v + 0.2, f'{v:.2f}', ha='center', fontweight='bold')

# 5b. Bias (mean residual) by artist tier — shows over/under prediction
colors_bias = ['coral' if b < 0 else 'steelblue' for b in tier_summary['Bias']]
axes[1].bar(tier_summary.index, tier_summary['Bias'], color=colors_bias, edgecolor='white', width=0.5)
axes[1].axhline(0, color='black', linewidth=1, linestyle='--')
axes[1].set_title('Prediction Bias by Artist Tier\n(negative = underpredict)', fontweight='bold')
axes[1].set_ylabel('Mean Residual (Actual − Predicted)')
axes[1].set_xlabel('Artist Tier')
axes[1].tick_params(axis='x', rotation=20)

# 5c. Boxplot of abs_error by actual popularity tier
order = ['Low (<40)', 'Mid (40–70)', 'High (>70)']
sns.boxplot(data=test_df, x='pop_tier', y='abs_error', ax=axes[2],
            order=order, color='steelblue',
            medianprops=dict(color='red', linewidth=2))
axes[2].set_title('Absolute Error by Actual Popularity Tier', fontweight='bold')
axes[2].set_xlabel('Actual Popularity Tier')
axes[2].set_ylabel('Absolute Error')
plt.suptitle('')

plt.tight_layout()
plt.savefig(OUT + '4_error_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  ✓ Saved: 4_error_analysis.png\n")


# ══════════════════════════════════════════════════════════════
#  SECTION 6 — DISTRIBUTION COMPARISON
# ══════════════════════════════════════════════════════════════
print("="*55)
print("  SECTION 6 — Distribution Comparison")
print("="*55)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Predicted vs Actual Popularity Distribution',
             fontsize=13, fontweight='bold')

# 6a. KDE overlay
sns.kdeplot(y_test.values,   label='Actual',             color='black',      linewidth=2.5, ax=axes[0])
sns.kdeplot(preds_audio,     label='Model 1 (Audio)',     color='steelblue',  linewidth=1.5,
            linestyle='--', ax=axes[0])
sns.kdeplot(preds_combined,  label='Model 2 (Combined)',  color='coral',      linewidth=1.5,
            linestyle='--', ax=axes[0])
axes[0].set_xlabel('Popularity Score (0–100)')
axes[0].set_ylabel('Density')
axes[0].set_title('KDE: Actual vs Predicted\n(good model ≈ overlaps with Actual)')
axes[0].legend()

# Annotate regression-to-mean effect
axes[0].annotate('Model 1 collapses\nto narrow range\n(regression-to-mean)',
                 xy=(50, axes[0].get_ylim()[1]*0.6),
                 fontsize=8, color='steelblue',
                 ha='center',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

# 6b. Box plots side by side
plot_data = pd.DataFrame({
    'Actual':           y_test.values,
    'Model 1\n(Audio)': preds_audio,
    'Model 2\n(Combined)': preds_combined
})
plot_data.boxplot(ax=axes[1],
                  boxprops=dict(color='steelblue'),
                  medianprops=dict(color='red', linewidth=2),
                  whiskerprops=dict(color='steelblue'),
                  capprops=dict(color='steelblue'))
axes[1].set_ylabel('Popularity Score')
axes[1].set_title('Distribution Summary: Boxplots\n(IQR, median, outliers)')

# Print summary stats
print("  ── Distribution Summary Statistics ──")
for label, arr in [('Actual', y_test.values),
                   ('Model 1 Predicted', preds_audio),
                   ('Model 2 Predicted', preds_combined)]:
    print(f"  {label:25s}  mean={arr.mean():.1f}  std={arr.std():.1f}  "
          f"min={arr.min():.1f}  max={arr.max():.1f}")

plt.tight_layout()
plt.savefig(OUT + '5_distribution_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  ✓ Saved: 5_distribution_comparison.png\n")


# ══════════════════════════════════════════════════════════════
#  DONE
# ══════════════════════════════════════════════════════════════
print("="*55)
print(f"  All plots saved to:\n  {OUT}")
print("="*55)