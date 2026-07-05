import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import os

processed_data_path = "../data/processed/"
raw_data_path = "../data/raw/"

df = pd.read_csv(f"{processed_data_path}merged_data.csv")

df['poverty_rate'] = pd.to_numeric(df['poverty_rate'], errors='coerce')
df['median_hhi'] = pd.to_numeric(df['median_hhi'], errors='coerce')
df['population'] = pd.to_numeric(df['population'], errors='coerce')
df['population'] = df['population'].fillna(df['population'].median())
df['total_fatalities'] = pd.to_numeric(df['total_fatalities'], errors='coerce').fillna(0)
df['total_crashes'] = pd.to_numeric(df['total_crashes'], errors='coerce').fillna(0)
df['traffic_fatality_rate'] = df['total_fatalities'] / df['population'] * 100_000

scaler = MinMaxScaler(feature_range=(0, 100))

_k = 10
_global_rate = df['total_fatalities'].sum() / df['population'].sum() * 100_000
_weight = df['total_crashes'] / (df['total_crashes'] + _k)
_shrunk_rate = _weight * df['traffic_fatality_rate'] + (1 - _weight) * _global_rate
df['traffic_score'] = 100 - scaler.fit_transform(np.log1p(_shrunk_rate.values).reshape(-1, 1)).flatten()

df['air_quality_score'] = scaler.fit_transform(-df['Median AQI'].to_numpy().reshape(-1, 1)).flatten()

health_keys = ['DEPRESSION', 'OBESITY', 'CSMOKING', 'MHLTH']
health_scaled = scaler.fit_transform(df[health_keys])
df['health_score'] = (100 - health_scaled).mean(axis=1)

df['poverty_rate'] = df['poverty_rate'].fillna(df['poverty_rate'].median())
df['median_hhi'] = df['median_hhi'].fillna(df['median_hhi'].median())
econ_scaled = scaler.fit_transform(df[['poverty_rate', 'median_hhi']])
econ_scaled[:, 0] = 100 - econ_scaled[:, 0]
df['economic_score'] = econ_scaled.mean(axis=1)

file_path = Path(f"{raw_data_path}tl_2023_us_county.shp")
if not file_path.is_file():
    counties = gpd.read_file("https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip")
    counties.to_file(f"{raw_data_path}tl_2023_us_county.shp")
else:
    counties = gpd.read_file(f"{raw_data_path}tl_2023_us_county.shp")

counties['fips'] = counties['GEOID'].astype(str).str.zfill(5)
counties['lat'] = counties['INTPTLAT'].astype(float)
counties['lon'] = counties['INTPTLON'].astype(float)
df['fips'] = df['fips'].astype(str).str.zfill(5)
df = df.merge(counties[['fips', 'lat', 'lon']], on='fips', how='left')

observed = df[df['air_quality_score'].notna()].copy()
missing = df[df['air_quality_score'].isna()].copy()
if len(missing) > 0:
    obs_coords = observed[['lat', 'lon']].values
    miss_coords = missing[['lat', 'lon']].values
    distances = cdist(miss_coords, obs_coords)
    distances = np.where(distances == 0, 1e-10, distances)
    weights = 1 / distances**2
    weights = weights / weights.sum(axis=1, keepdims=True)
    df.loc[df['air_quality_score'].isna(), 'air_quality_score'] = weights @ observed['air_quality_score'].values

for col in ['health_score', 'traffic_score', 'air_quality_score', 'economic_score']:
    df[col] = df[col].fillna(df[col].median())

# ── Crime sub-score (CHR 2023: Homicides + Firearm Fatalities per 100k) ──────
chr_df = pd.read_csv(f"{raw_data_path}chr_2023.csv", low_memory=False, dtype=str)
chr_df = chr_df[chr_df['5-digit FIPS Code'] != 'fipscode'].copy()
chr_df = chr_df[~chr_df['5-digit FIPS Code'].str.endswith('000')].copy()
chr_df = chr_df[chr_df['5-digit FIPS Code'].str.len() == 5].copy()
chr_df['fips'] = chr_df['5-digit FIPS Code'].str.zfill(5)
chr_df['homicide_rate'] = pd.to_numeric(chr_df['Homicides raw value'], errors='coerce')
chr_df['firearm_rate'] = pd.to_numeric(chr_df['Firearm Fatalities raw value'], errors='coerce')

df = df.merge(chr_df[['fips', 'homicide_rate', 'firearm_rate']], on='fips', how='left')

# Impute missing with state median, then national median fallback.
# Homicide rate has ~42% coverage; firearm has ~72%.
for col in ['homicide_rate', 'firearm_rate']:
    state_med = df.groupby('state_abbr')[col].transform('median')
    national_med = df[col].median()
    df[col] = df[col].fillna(state_med).fillna(national_med)

homicide_scaled = 100 - scaler.fit_transform(df['homicide_rate'].values.reshape(-1, 1)).flatten()
firearm_scaled = 100 - scaler.fit_transform(df['firearm_rate'].values.reshape(-1, 1)).flatten()
df['crime_score'] = 0.6 * homicide_scaled + 0.4 * firearm_scaled
print(f"Crime score: mean={df['crime_score'].mean():.1f}, std={df['crime_score'].std():.1f}")

WEIGHTS = {
    'health_score':      0.20,
    'air_quality_score': 0.20,
    'economic_score':    0.20,
    'traffic_score':     0.20,
    'crime_score':       0.20,
}
df['safety_score'] = sum(w * df[col] for col, w in WEIGHTS.items())

df.to_csv(f"{processed_data_path}features.csv", index=False)
print(f"Saved features.csv with shape {df.shape}")
print(df[['name','safety_score']].sort_values('safety_score').head(10))
