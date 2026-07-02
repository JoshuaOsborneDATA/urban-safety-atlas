import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import os

processed_data_path = "../data/processed/"
eda_figure_folder = "../figs/eda/"
os.makedirs(eda_figure_folder, exist_ok=True)

df = pd.read_csv(f"{processed_data_path}merged_data.csv")
df['poverty_rate'] = df['poverty_rate'].replace('.', float('nan')).astype(float)
df['median_hhi'] = df['median_hhi'].replace('.', float('nan')).astype(float)

for key in df.select_dtypes(include='number').columns:
    plt.figure()
    sns.kdeplot(df, x=key)
    plt.savefig(f"{eda_figure_folder}kde_{key}.png")
    plt.close()

plt.figure(figsize=(40, 20))
sns.heatmap(df.corr(method='pearson', numeric_only=True), annot=True, cmap='jet', fmt='.2f')
plt.savefig(eda_figure_folder + "heatmap.png")
plt.close()

print(df.groupby('census_region')['Median AQI'].describe())

def get_pearsonr_no_nan(df, x, y):
    mask = df[x].notna() & df[y].notna()
    r, p = scipy.stats.pearsonr(df.loc[mask, x].astype(float), df.loc[mask, y])
    return r, p

def get_spearmanr_no_nan(df, x, y):
    mask = df[x].notna() & df[y].notna()
    r, p = scipy.stats.spearmanr(df.loc[mask, x].astype(float), df.loc[mask, y])
    return r, p

df['traffic_fatality_rate'] = df['total_fatalities'] / df['population'] * 100000
r, p = get_pearsonr_no_nan(df, 'poverty_rate', 'traffic_fatality_rate')
print(f"Pearson poverty vs traffic fatality: r={r:.4f}, p={p:.4f}")
r, p = get_spearmanr_no_nan(df, 'poverty_rate', 'traffic_fatality_rate')
print(f"Spearman poverty vs traffic fatality: r={r:.4f}, p={p:.4f}")
print("EDA complete.")
