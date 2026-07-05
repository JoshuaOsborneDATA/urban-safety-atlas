import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

processed_data_path = "../data/processed/"
fig_dir = "../figs/"
os.makedirs(fig_dir, exist_ok=True)

df = pd.read_csv(f"{processed_data_path}features.csv")

score_cols = ["health_score", "air_quality_score", "economic_score", "traffic_score", "crime_score"]

standard_scale = StandardScaler()
X_scaled = standard_scale.fit_transform(df[score_cols])

pca = PCA(n_components=2)
pca.fit(X_scaled)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("PCA loadings:\n", pca.components_)

def add_watermark(fig):
    fig.text(0.99, 0.01, '© Joshua Osborne', ha='right', va='bottom',
             fontsize=7, color='gray', alpha=0.6, transform=fig.transFigure)

fig = plt.figure()
plt.title('PCA Loadings')
plt.scatter(*pca.components_)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
add_watermark(fig)
plt.savefig(f'{fig_dir}pca.png')
plt.close()

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_coords = tsne.fit_transform(X_scaled)

fig = plt.figure()
plt.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=df['safety_score'], cmap='RdYlGn')
plt.colorbar(label='safety_score')
plt.title('t-SNE colored by safety_score')
add_watermark(fig)
plt.savefig(f'{fig_dir}tsne.png')
plt.close()

np.save(f"{processed_data_path}tsne_coords.npy", tsne_coords)
print("Saved tsne_coords.npy")
