import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors

processed_data_path = "../data/processed/"
df = pd.read_csv(f"{processed_data_path}final_data.csv")

# ── Shapiro-Wilk normality test ───────────────────────────────────────────────
print("Shapiro-Wilk normality test per region:")
for region, group in df.groupby('census_region'):
    sample = group['safety_score'].dropna()
    if len(sample) > 500:
        sample = sample.sample(500, random_state=42)
    stat, p = stats.shapiro(sample)
    print(f"  {region}: W={stat:.4f}, p={p:.4f}")

# ── Kruskal-Wallis test ───────────────────────────────────────────────────────
groups = [group['safety_score'].dropna().values for _, group in df.groupby('census_region')]
stat, p = stats.kruskal(*groups)
print(f"\nKruskal-Wallis: H={stat:.4f}, p={p:.4f}")

print("\nMedian safety_score by region:")
print(df.groupby('census_region')['safety_score'].median().sort_values())

# ── Global Moran's I (spatial autocorrelation) ────────────────────────────────
print("\nGlobal Moran's I:")
valid = df.dropna(subset=['lat', 'lon', 'safety_score']).copy()
coords = valid[['lat', 'lon']].values

nbrs = NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(coords)
_, indices = nbrs.kneighbors(coords)
neighbor_idx = indices[:, 1:]  # drop self-reference (first column)

x = valid['safety_score'].values
x_d = x - x.mean()
# Row-standardised spatial lag = mean of k nearest neighbours
spatial_lag = np.mean(x_d[neighbor_idx], axis=1)
# Moran's I = sum(z * lag) / sum(z^2)  [row-standardised weights, W = n]
I = float(np.sum(x_d * spatial_lag) / np.sum(x_d**2))
EI = -1.0 / (len(x) - 1)
z_approx = (I - EI) / (1.0 / np.sqrt(len(x)))

print(f"  Global Moran's I = {I:.4f}")
print(f"  Expected I (random) = {EI:.6f}")
print(f"  Z-score (approx)    = {z_approx:.2f}")
print(f"  Interpretation: {'Strong' if I > 0.5 else 'Moderate' if I > 0.2 else 'Weak'} positive spatial autocorrelation")
print(f"  (k=8 nearest county centroids, row-standardised weights, n={len(x)})")

# LISA quadrant counts
x_std = (x - x.mean()) / x.std()
lag_std = (spatial_lag - spatial_lag.mean()) / spatial_lag.std()
hh = int(((x_std >= 0) & (lag_std >= 0)).sum())
ll = int(((x_std < 0) & (lag_std < 0)).sum())
hl = int(((x_std >= 0) & (lag_std < 0)).sum())
lh = int(((x_std < 0) & (lag_std >= 0)).sum())
print(f"\n  LISA quadrant counts:")
print(f"    HH (safe surrounded by safe):    {hh}")
print(f"    LL (unsafe surrounded by unsafe): {ll}")
print(f"    HL (safe island in danger region): {hl}")
print(f"    LH (danger pocket in safe region): {lh}")
