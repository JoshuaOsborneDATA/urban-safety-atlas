import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score
import os

processed_data_path = "../data/processed/"
model_dir = "../models/"
fig_dir = "../figs/"
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

df = pd.read_csv(f"{processed_data_path}features.csv")
tsne_coords = np.load(f"{processed_data_path}tsne_coords.npy")

score_cols = ["health_score", "air_quality_score", "economic_score", "traffic_score", "crime_score"]

standard_scale = StandardScaler()
X_scaled = standard_scale.fit_transform(df[score_cols])

random_state = 42
inertias, silhouettes = [], []
k_range = range(2, 50)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

def add_watermark(fig):
    fig.text(0.99, 0.01, '© Joshua Osborne', ha='right', va='bottom',
             fontsize=7, color='gray', alpha=0.6, transform=fig.transFigure)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(k_range), inertias, marker='o', markersize=3)
axes[0].set_title('Inertia vs k')
axes[0].set_xlabel('k')
axes[0].set_ylabel('Inertia')
axes[1].plot(list(k_range), silhouettes, marker='o', markersize=3)
axes[1].set_title('Silhouette Score vs k')
axes[1].set_xlabel('k')
axes[1].set_ylabel('Silhouette Score')
plt.tight_layout()
add_watermark(fig)
plt.savefig(f'{fig_dir}elbow_silhouette.png', dpi=120)
plt.close()

km_final = KMeans(n_clusters=2, random_state=random_state, n_init='auto')
df['KM_cluster'] = km_final.fit_predict(X_scaled)
print("KMeans mean safety_score per cluster:")
print(df.groupby('KM_cluster')['safety_score'].mean().sort_values(ascending=False))

hdb = HDBSCAN(min_cluster_size=15, min_samples=5)
df['HDB_cluster'] = hdb.fit_predict(X_scaled)
print("\nHDBSCAN cluster counts:")
print(df['HDB_cluster'].value_counts())
print("\nHDBSCAN mean safety_score per cluster:")
print(df.groupby('HDB_cluster')['safety_score'].mean().sort_values())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cmap_km = plt.cm.get_cmap('tab10', df['KM_cluster'].nunique())
scatter1 = axes[0].scatter(tsne_coords[:,0], tsne_coords[:,1], c=df['KM_cluster'], cmap=cmap_km, s=5, alpha=0.6)
axes[0].set_title('KMeans (k=2)')
axes[0].set_xlabel('t-SNE 1')
axes[0].set_ylabel('t-SNE 2')
plt.colorbar(scatter1, ax=axes[0])

cmap_hdb = plt.cm.get_cmap('tab10', df['HDB_cluster'].nunique())
scatter2 = axes[1].scatter(tsne_coords[:,0], tsne_coords[:,1], c=df['HDB_cluster'], cmap=cmap_hdb, s=5, alpha=0.6)
axes[1].set_title('HDBSCAN')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
plt.colorbar(scatter2, ax=axes[1])
plt.tight_layout()
add_watermark(fig)
plt.savefig(f'{fig_dir}cluster_comparison.png', dpi=120)
plt.close()

df.to_csv(f"{processed_data_path}final_data.csv", index=False)
print(f"Saved final_data.csv with shape {df.shape}")

joblib.dump(standard_scale, f"{model_dir}scaler.pkl")
joblib.dump(km_final,       f"{model_dir}kmeans_k2.pkl")
joblib.dump(hdb,            f"{model_dir}hdbscan.pkl")
print(f"Saved scaler, kmeans_k2, hdbscan to {model_dir}")
