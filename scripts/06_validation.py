import pandas as pd
from scipy.stats import pearsonr, spearmanr

processed_data_path = "../data/processed/"

df = pd.read_csv(f"{processed_data_path}final_data.csv")
le = pd.read_csv(f"{processed_data_path}life_expectancy.csv", dtype={'fips': str})
# CHR small-population counties can produce implausible estimates; >100 is biologically impossible
le.loc[le['life_expectancy'] > 100, 'life_expectancy'] = float('nan')

SCORE_COLS = ['health_score', 'air_quality_score', 'economic_score', 'traffic_score']
SCORE_LABELS = ['Health', 'Air Quality', 'Economic', 'Traffic']

df['fips'] = df['fips'].astype(str).str.zfill(5)
merged = df.merge(le, on='fips', how='inner').dropna(subset=['safety_score', 'life_expectancy'])
print(f"Matched counties: {len(merged)}")

# ── Overall correlation ───────────────────────────────────────────────────────
rho, p_rho = spearmanr(merged['safety_score'], merged['life_expectancy'])
r,   p_r   = pearsonr(merged['safety_score'],  merged['life_expectancy'])
print(f"\nSafety score vs Life Expectancy (CHR 2023):")
print(f"  Spearman rho = {rho:.4f}  (p = {p_rho:.2e})")
print(f"  Pearson   r  = {r:.4f}   (p = {p_r:.2e})")

# ── By sub-score ──────────────────────────────────────────────────────────────
print(f"\nSub-score correlations with life expectancy:")
for col, lbl in zip(SCORE_COLS, SCORE_LABELS):
    sub = merged[[col, 'life_expectancy']].dropna()
    rho_s, p_s = spearmanr(sub[col], sub['life_expectancy'])
    print(f"  {lbl:<14} rho = {rho_s:+.4f}  (p = {p_s:.2e})")

# ── Additional outcomes ───────────────────────────────────────────────────────
print(f"\nAdditional external outcomes (safety_score, expect negative rho):")
for col, lbl in [
    ('premature_mortality', 'Premature Mortality'),
    ('drug_overdose_deaths', 'Drug Overdose Deaths'),
    ('child_mortality',      'Child Mortality'),
]:
    if col not in merged.columns:
        continue
    sub = merged[['safety_score', col]].dropna()
    rho_b, p_b = spearmanr(sub['safety_score'], sub[col])
    direction = 'PASS' if rho_b < 0 else 'FAIL'
    print(f"  {lbl:<26} rho = {rho_b:+.4f}  [{direction}]")
