import json
import os
from urllib.request import urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data/processed/final_data.csv")
LE_PATH = os.path.join(BASE_DIR, "data/processed/life_expectancy.csv")
TSNE_PATH = os.path.join(BASE_DIR, "data/processed/tsne_coords.npy")

SCORE_COLS = ["health_score", "air_quality_score", "economic_score", "traffic_score", "crime_score"]
SCORE_LABELS = ["Health", "Air Quality", "Economic", "Traffic", "Crime"]

WEIGHT_PRESETS = {
    "Equal (Recommended)": [0.20, 0.20, 0.20, 0.20, 0.20],
    "Health-Heavy":        [0.40, 0.15, 0.15, 0.15, 0.15],
    "Traffic-Heavy":       [0.15, 0.15, 0.15, 0.40, 0.15],
    "Economic-Heavy":      [0.15, 0.15, 0.40, 0.15, 0.15],
    "Crime-Heavy":         [0.15, 0.15, 0.15, 0.15, 0.40],
    "Custom":              None,
}


# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    for col in SCORE_COLS:
        df[col] = df[col].fillna(df[col].median())
    df["safety_score"] = df[SCORE_COLS].mean(axis=1)
    return df


@st.cache_data
def load_geojson() -> dict:
    with urlopen(
        "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    ) as r:
        return json.load(r)


@st.cache_data
def load_life_expectancy() -> pd.DataFrame:
    df = pd.read_csv(LE_PATH, dtype={"fips": str})
    # CHR small-population counties can produce implausible estimates; >100 is biologically impossible
    # for a county average (e.g. Aleutians East Borough CI spans 68–157 years)
    df.loc[df["life_expectancy"] > 100, "life_expectancy"] = float("nan")
    return df


# ── Clustering (cached per params so sliders don't re-run unnecessarily) ─────

@st.cache_data
def compute_validation_data(weights: tuple, traffic_method: str, bayes_k: int) -> dict:
    from scipy.stats import spearmanr, pearsonr

    df = load_data().copy()
    if traffic_method == "Bayesian Shrinkage":
        df["traffic_score"] = compute_traffic_score_bayes(bayes_k).values
    df["safety_score"] = sum(w * df[col] for w, col in zip(weights, SCORE_COLS))

    le_df = load_life_expectancy()
    merged = df.merge(le_df, on="fips", how="inner").dropna(subset=["safety_score", "life_expectancy"])

    rho, p_rho = spearmanr(merged["safety_score"], merged["life_expectancy"])
    r, p_r = pearsonr(merged["safety_score"], merged["life_expectancy"])

    sub_corrs = []
    for col, lbl in zip(SCORE_COLS, SCORE_LABELS):
        sub = merged[[col, "life_expectancy"]].dropna()
        rho_s, _ = spearmanr(sub[col], sub["life_expectancy"])
        sub_corrs.append({"Sub-score": lbl, "Spearman ρ": rho_s, "n": len(sub)})

    outcome_cols = {
        "premature_mortality": "Premature Mortality",
        "drug_overdose_deaths": "Drug Overdose Deaths",
        "child_mortality": "Child Mortality",
    }
    available_outcomes = {
        col: lbl for col, lbl in outcome_cols.items()
        if col in merged.columns and merged[col].dropna().shape[0] >= 100
    }

    bench_rows = []
    for col, lbl in available_outcomes.items():
        sub = merged[["safety_score", col]].dropna()
        rho_b, p_b = spearmanr(sub["safety_score"], sub[col])
        bench_rows.append({
            "Outcome": lbl,
            "Spearman ρ": round(rho_b, 4),
            "p-value": "< 0.001" if p_b < 0.001 else f"{p_b:.3e}",
            "n": len(sub),
            "Expected direction": "(−) higher safety → lower mortality",
        })

    all_outcomes = {"Life Expectancy": "life_expectancy"} | {lbl: col for col, lbl in available_outcomes.items()}
    heatmap_data = {}
    for out_lbl, out_col in all_outcomes.items():
        col_rhos = []
        for score_col in SCORE_COLS:
            sub = merged[[score_col, out_col]].dropna()
            rho_s, _ = spearmanr(sub[score_col], sub[out_col])
            col_rhos.append(round(rho_s, 3))
        heatmap_data[out_lbl] = col_rhos

    return {
        "merged": merged,
        "rho": rho, "p_rho": p_rho,
        "r": r, "p_r": p_r,
        "sub_corrs": sub_corrs,
        "bench_rows": bench_rows,
        "heatmap_data": heatmap_data,
        "available_outcomes": available_outcomes,
    }


@st.cache_data
def compute_morans_cached(weights: tuple, traffic_method: str, bayes_k: int, knn_k: int = 8):
    df = load_data().copy()
    if traffic_method == "Bayesian Shrinkage":
        df["traffic_score"] = compute_traffic_score_bayes(bayes_k).values
    df["safety_score"] = sum(w * df[col] for w, col in zip(weights, SCORE_COLS))
    valid_idx, neighbor_indices = get_knn_indices(k=knn_k)
    safety_vals = df.loc[valid_idx, "safety_score"].values
    morans_I, EI, spatial_lag = compute_morans_i(safety_vals, neighbor_indices)
    names = df.loc[valid_idx, "name"].values
    return morans_I, EI, spatial_lag, safety_vals, valid_idx, names


@st.cache_data
def load_tsne() -> np.ndarray:
    return np.load(TSNE_PATH)


@st.cache_data
def compute_traffic_score_bayes(k: int) -> pd.Series:
    """Return Bayesian-shrunk traffic score (0-100, higher = safer)."""
    df = load_data()
    global_rate = df["total_fatalities"].sum() / df["population"].sum() * 100_000
    weight = df["total_crashes"] / (df["total_crashes"] + k)
    shrunk_rate = weight * df["traffic_fatality_rate"] + (1 - weight) * global_rate
    score = 100 - MinMaxScaler(feature_range=(0, 100)).fit_transform(
        np.log1p(shrunk_rate.values).reshape(-1, 1)
    ).flatten()
    return pd.Series(score, index=df.index)


@st.cache_data
def run_kmeans(k: int) -> np.ndarray:
    df = load_data()
    X = StandardScaler().fit_transform(df[SCORE_COLS].values)
    return KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X)


@st.cache_data
def run_hdbscan(min_cluster_size: int, min_samples: int) -> np.ndarray:
    df = load_data()
    X = StandardScaler().fit_transform(df[SCORE_COLS].values)
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(X)


# ── Spatial weights (KNN indices, depends only on lat/lon) ──────────────────

@st.cache_data
def get_knn_indices(k: int = 8):
    """Return (row_indices, neighbor_index_array) for spatial weight matrix."""
    df = load_data()
    valid = df.dropna(subset=["lat", "lon"]).copy()
    coords = valid[["lat", "lon"]].values
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
    _, indices = nbrs.kneighbors(coords)
    return valid.index.to_numpy(), indices[:, 1:]  # drop self


def compute_morans_i(values: np.ndarray, neighbor_indices: np.ndarray):
    """Compute Global Moran's I using row-standardised KNN weights.

    Row-standardised means each row sums to 1, so W = n and I = sum(z*lag)/sum(z^2).
    """
    x = values - values.mean()
    spatial_lag = np.mean(x[neighbor_indices], axis=1)
    I = float(np.sum(x * spatial_lag) / np.sum(x**2))
    EI = -1.0 / (len(x) - 1)
    return I, EI, spatial_lag


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Urban Safety Atlas",
    page_icon="🗺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Urban Safety Atlas")
st.caption("County-level US safety index · EPA air quality · CDC PLACES health · NHTSA FARS traffic · Census SAIPE poverty · CHR 2023 crime")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Filters & Controls")

    # Region
    df_full = load_data()
    regions = ["All"] + sorted(df_full["census_region"].dropna().unique().tolist())
    selected_region = st.selectbox("Census Region", regions)

    st.divider()

    # Weight controls
    st.subheader("Sub-score Weights")
    preset_name = st.selectbox("Weight Preset", list(WEIGHT_PRESETS.keys()))

    if preset_name == "Custom":
        st.markdown("**Drag to set weights** (auto-normalised to sum to 1)")
        w_health = st.slider("Health",       0.0, 1.0, 0.20, 0.05)
        w_air    = st.slider("Air Quality",  0.0, 1.0, 0.20, 0.05)
        w_econ   = st.slider("Economic",     0.0, 1.0, 0.20, 0.05)
        w_traf   = st.slider("Traffic",      0.0, 1.0, 0.20, 0.05)
        w_crime  = st.slider("Crime",        0.0, 1.0, 0.20, 0.05)
        raw_weights = [w_health, w_air, w_econ, w_traf, w_crime]
        total_w = sum(raw_weights) or 1.0
        if abs(total_w - 1.0) > 0.01:
            st.caption(f"Weights sum to {total_w:.2f}; normalising.")
        weights = [w / total_w for w in raw_weights]
    else:
        weights = WEIGHT_PRESETS[preset_name]
        st.caption(
            " · ".join(f"{lbl}: {w:.2f}" for lbl, w in zip(SCORE_LABELS, weights))
        )

    st.divider()

    # Clustering controls
    st.subheader("Clustering")
    cluster_method = st.selectbox("Method", ["KMeans", "HDBSCAN"])

    if cluster_method == "KMeans":
        km_k = st.slider("Number of clusters (k)", 2, 10, 2)
    else:
        hdb_min_cs = st.slider("min_cluster_size", 5, 80, 15, 5)
        hdb_min_s  = st.slider("min_samples",      1, 20,  5, 1)

    st.divider()

    # Traffic score method
    st.subheader("Traffic Score Method")
    traffic_method = st.radio(
        "Scoring variant",
        ["Bayesian Shrinkage", "Log"],
        help=(
            "Log: log1p-transforms the raw fatality rate before scaling. "
            "Bayesian Shrinkage: pulls each county's rate toward the national mean "
            "weighted by the number of crashes — useful for small-population counties "
            "but compresses county-to-county variation."
        ),
    )
    if traffic_method == "Bayesian Shrinkage":
        bayes_k = st.slider(
            "Smoothing factor k",
            min_value=1, max_value=100, value=10,
            help="Higher k means a stronger pull toward the national mean. k=10 is the domain default: a county needs at least 10 crashes before its observed rate is trusted over the prior.",
        )


# ── Derived data ──────────────────────────────────────────────────────────────

# Optionally swap traffic_score with Bayesian variant before weighting
df_full = df_full.copy()
if traffic_method == "Bayesian Shrinkage":
    df_full["traffic_score"] = compute_traffic_score_bayes(bayes_k).values

# Recompute safety_score with current weights
df_full["safety_score"] = sum(w * df_full[col] for w, col in zip(weights, SCORE_COLS))

# Attach cluster labels
if cluster_method == "KMeans":
    cluster_labels = run_kmeans(km_k)
    df_full["cluster"] = cluster_labels
    # Label clusters by mean safety_score (descending = safer)
    mean_by_cluster = df_full.groupby("cluster")["safety_score"].mean().sort_values(ascending=False)
    tier_names = {c: f"Tier {i+1} ({'Safer' if i==0 else 'More Dangerous' if i==len(mean_by_cluster)-1 else 'Moderate'})"
                  for i, c in enumerate(mean_by_cluster.index)}
    df_full["cluster_label"] = df_full["cluster"].map(tier_names)
else:
    cluster_labels = run_hdbscan(hdb_min_cs, hdb_min_s)
    df_full["cluster"] = cluster_labels
    df_full["cluster_label"] = df_full["cluster"].apply(
        lambda x: "Noise" if x == -1 else f"Cluster {x}"
    )

# Apply region filter
df = df_full[df_full["census_region"] == selected_region].copy() if selected_region != "All" else df_full.copy()

# ── Metrics row ───────────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
col1.metric("Counties", f"{len(df):,}")

valid_df = df.dropna(subset=["safety_score"])
if len(valid_df):
    safest = valid_df.loc[valid_df["safety_score"].idxmax()]
    danger = valid_df.loc[valid_df["safety_score"].idxmin()]
    col2.metric("Safest County", safest["name"])
    col3.metric("Most Dangerous", danger["name"])
    col4.metric("Median Score", f"{valid_df['safety_score'].median():.1f}")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_map, tab_eda, tab_clusters, tab_profile, tab_spatial, tab_valid = st.tabs(
    ["Map", "EDA", "Clusters", "County Profile", "Spatial", "Validation"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: MAP
# ══════════════════════════════════════════════════════════════════════════════

with tab_map:
    st.subheader("Safety Score by County")
    w_str = " · ".join(f"{lbl} {w:.2f}" for lbl, w in zip(SCORE_LABELS, weights))
    st.caption(f"Weights: {w_str}")

    if len(df) == 0:
        st.info("No counties match the current filter.")
    else:
        counties_geo = load_geojson()
        fig_map = px.choropleth(
            df,
            geojson=counties_geo,
            locations="fips",
            color="safety_score",
            color_continuous_scale="RdYlGn",
            range_color=(0, 100),
            scope="usa",
            hover_name="name",
            hover_data={
                "safety_score": ":.1f",
                "health_score": ":.1f",
                "air_quality_score": ":.1f",
                "traffic_score": ":.1f",
                "economic_score": ":.1f",
                "crime_score": ":.1f",
                "fips": False,
            },
            title="County Safety Score (weighted composite)",
        )
        fig_map.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=520)
        st.plotly_chart(fig_map, use_container_width=True)

        col_safe, col_danger = st.columns(2)
        with col_safe:
            st.markdown("**Top 10 Safest**")
            st.dataframe(
                df.nlargest(10, "safety_score")[["name", "safety_score", "census_region"]]
                .round(1).reset_index(drop=True),
                use_container_width=True,
            )
        with col_danger:
            st.markdown("**Bottom 10 Most Dangerous**")
            st.dataframe(
                df.nsmallest(10, "safety_score")[["name", "safety_score", "census_region"]]
                .round(1).reset_index(drop=True),
                use_container_width=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: EDA
# ══════════════════════════════════════════════════════════════════════════════

with tab_eda:
    st.subheader("Exploratory Data Analysis")

    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist = px.histogram(
            df_full, x="safety_score", color="census_region", nbins=40,
            facet_col="census_region",
            title="Safety Score Distribution by Region",
            labels={"safety_score": "Safety Score (0-100)"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        df_sc = df.dropna(subset=["traffic_fatality_rate", "poverty_rate"])
        df_sc = df_sc[df_sc["traffic_fatality_rate"] < 500]
        fig_scatter = px.scatter(
            df_sc, x="poverty_rate", y="traffic_fatality_rate",
            color="safety_score", color_continuous_scale="RdYlGn",
            hover_name="name",
            title="Traffic Fatality Rate vs Poverty Rate",
            labels={
                "poverty_rate": "Poverty Rate (%)",
                "traffic_fatality_rate": "Traffic Fatality Rate (per 100k)",
            },
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Sub-score correlation heatmap
    corr = df[SCORE_COLS + ["safety_score"]].corr().round(2)
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text}", showscale=True,
    ))
    fig_corr.update_layout(title="Sub-score Correlation Matrix", height=400, xaxis={"tickangle": -20})
    st.plotly_chart(fig_corr, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: CLUSTERS
# ══════════════════════════════════════════════════════════════════════════════

with tab_clusters:
    if cluster_method == "KMeans":
        st.subheader(f"KMeans Clustering  (k = {km_k})")
    else:
        st.subheader(f"HDBSCAN Clustering  (min_cluster_size={hdb_min_cs}, min_samples={hdb_min_s})")

    # Cluster summary metrics
    cluster_counts = df["cluster_label"].value_counts()
    n_clusters = df["cluster"].nunique()
    noise_count = int((df["cluster"] == -1).sum()) if cluster_method == "HDBSCAN" else 0

    mcols = st.columns(min(n_clusters + (1 if noise_count else 0), 5))
    for i, (label, cnt) in enumerate(cluster_counts.items()):
        if i < len(mcols):
            mcols[i].metric(label, f"{cnt} counties")

    # Bar chart: mean sub-scores per cluster
    cluster_means = (
        df.groupby("cluster_label")[SCORE_COLS]
        .mean()
        .reset_index()
    )
    cluster_long = cluster_means.melt(
        id_vars="cluster_label", var_name="sub_score", value_name="mean_score"
    )
    cluster_long["sub_score"] = cluster_long["sub_score"].str.replace("_score", "").str.title()

    fig_bar = px.bar(
        cluster_long, x="cluster_label", y="mean_score", color="sub_score",
        barmode="group",
        title=f"Mean Sub-Scores per {cluster_method} Cluster",
        labels={"cluster_label": "Cluster", "mean_score": "Mean Score (0-100)"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Safety score distribution per cluster
    fig_box = px.box(
        df[df["cluster"] != -1], x="cluster_label", y="safety_score",
        color="cluster_label",
        title=f"Safety Score Distribution by Cluster",
        labels={"cluster_label": "Cluster", "safety_score": "Safety Score"},
    )
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    # t-SNE coloured by cluster
    tsne_coords = load_tsne()
    tsne_df = pd.DataFrame({
        "t-SNE 1": tsne_coords[:, 0],
        "t-SNE 2": tsne_coords[:, 1],
        "cluster_label": df_full["cluster_label"].values,
        "name": df_full["name"].values,
        "safety_score": df_full["safety_score"].values,
    })
    fig_tsne = px.scatter(
        tsne_df,
        x="t-SNE 1", y="t-SNE 2",
        color="cluster_label",
        hover_name="name",
        hover_data={"safety_score": ":.1f", "t-SNE 1": False, "t-SNE 2": False},
        opacity=0.6,
        title=f"t-SNE: {cluster_method} Cluster Structure",
        labels={"cluster_label": "Cluster"},
    )
    fig_tsne.update_traces(marker_size=4)
    fig_tsne.update_layout(height=480)
    st.plotly_chart(fig_tsne, use_container_width=True)

    if cluster_method == "HDBSCAN" and noise_count > 0:
        st.info(
            f"{noise_count} counties classified as noise (cluster = -1); these are outliers "
            "that don't fit any cluster. Try increasing min_cluster_size to reduce noise."
        )

    col_top, col_bot = st.columns(2)
    with col_top:
        st.markdown("**Top 10 Safest Counties**")
        st.dataframe(
            df.nlargest(10, "safety_score")[["name", "safety_score", "cluster_label", "census_region"]]
            .round(1).reset_index(drop=True),
            use_container_width=True,
        )
    with col_bot:
        st.markdown("**Bottom 10 Most Dangerous Counties**")
        st.dataframe(
            df.nsmallest(10, "safety_score")[["name", "safety_score", "cluster_label", "census_region"]]
            .round(1).reset_index(drop=True),
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: COUNTY PROFILE
# ══════════════════════════════════════════════════════════════════════════════

with tab_profile:
    st.subheader("County Profile")

    county_names = sorted(df["name"].dropna().unique().tolist())
    if len(county_names) == 0:
        st.info("No counties available with current filter.")
    else:
        selected_county = st.selectbox("Select a County", county_names)
        county_data = df[df["name"] == selected_county].iloc[0]
        national_median = df_full[SCORE_COLS + ["safety_score"]].median()

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Safety Score", f"{county_data['safety_score']:.1f}")
        c2.metric("Health", f"{county_data['health_score']:.1f}")
        c3.metric("Air Quality", f"{county_data['air_quality_score']:.1f}")
        c4.metric("Traffic", f"{county_data['traffic_score']:.1f}")
        c5.metric("Economic", f"{county_data['economic_score']:.1f}")
        c6.metric("Crime", f"{county_data['crime_score']:.1f}")

        county_vals = [county_data[col] for col in SCORE_COLS]
        median_vals = [national_median[col] for col in SCORE_COLS]
        categories = SCORE_LABELS

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=county_vals + [county_vals[0]],
            theta=categories + [categories[0]],
            fill="toself", name=selected_county,
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=median_vals + [median_vals[0]],
            theta=categories + [categories[0]],
            fill="toself", name="National Median", opacity=0.5,
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title=f"{selected_county} vs National Median",
            height=450,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("**Raw indicators**")
        col_l, col_r = st.columns(2)
        with col_l:
            st.write(f"Poverty Rate: **{county_data['poverty_rate']:.1f}%**")
            st.write(f"Median HHI: **${county_data['median_hhi']:,.0f}**")
            st.write(f"Depression: **{county_data['DEPRESSION']:.1f}%**")
            st.write(f"Obesity: **{county_data['OBESITY']:.1f}%**")
        with col_r:
            st.write(f"Smoking: **{county_data['CSMOKING']:.1f}%**")
            st.write(f"Mental Health (MHLTH): **{county_data['MHLTH']:.1f}%**")
            tfr = county_data.get("traffic_fatality_rate", float("nan"))
            st.write(f"Traffic Fatality Rate: **{tfr:.1f} per 100k**")
            st.write(f"Homicide Rate: **{county_data.get('homicide_rate', float('nan')):.1f} per 100k**")
            st.write(f"Firearm Fatality Rate: **{county_data.get('firearm_rate', float('nan')):.1f} per 100k**")
            st.write(f"Cluster: **{county_data['cluster_label']}**")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: SPATIAL (Moran's I)
# ══════════════════════════════════════════════════════════════════════════════

with tab_spatial:
    st.subheader("Spatial Autocorrelation: Global Moran's I")
    st.markdown(
        "Moran's I tests whether similar counties cluster geographically. "
        "I = 1 means perfect clustering, I = 0 means random, I = -1 means perfect dispersion. "
        "The spatial weights use the **8 nearest county centroids** (row-standardised)."
    )

    morans_I, EI, spatial_lag, safety_vals, valid_idx, _names = compute_morans_cached(
        tuple(weights), traffic_method, bayes_k if traffic_method == "Bayesian Shrinkage" else 10
    )

    # Z-score (simplified analytical approximation)
    n = len(safety_vals)
    z_approx = (morans_I - EI) / (1.0 / np.sqrt(n))  # rough std dev

    # Interpretation
    if morans_I > 0.5:
        interpretation = "Strong positive spatial autocorrelation: safe and unsafe counties strongly cluster geographically."
    elif morans_I > 0.2:
        interpretation = "Moderate positive spatial autocorrelation: neighbouring counties tend to have similar safety scores."
    elif morans_I > 0:
        interpretation = "Weak positive spatial autocorrelation: slight geographic tendency but largely random."
    else:
        interpretation = "Negative or near-zero spatial autocorrelation: safety scores are dispersed across space."

    m1, m2, m3 = st.columns(3)
    m1.metric("Global Moran's I", f"{morans_I:.4f}")
    m2.metric("Expected I (random)", f"{EI:.4f}")
    m3.metric("Z-score (approx)", f"{z_approx:.2f}")
    st.info(interpretation)
    st.caption(
        f"Computed using k=8 nearest county centroids (row-standardised). "
        f"n = {n} counties. Current weight scheme: {preset_name}."
    )

    # Moran scatter plot: standardised score vs spatially-lagged score
    x_std = (safety_vals - safety_vals.mean()) / safety_vals.std()
    lag_std = (spatial_lag - spatial_lag.mean()) / spatial_lag.std()

    # Quadrant classification
    quadrant = np.where(
        (x_std >= 0) & (lag_std >= 0), "HH (High-High)",
        np.where(
            (x_std < 0) & (lag_std < 0), "LL (Low-Low)",
            np.where((x_std >= 0) & (lag_std < 0), "HL (High-Low)", "LH (Low-High)"),
        ),
    )

    scatter_df = pd.DataFrame({
        "safety_std": x_std,
        "spatial_lag_std": lag_std,
        "quadrant": quadrant,
        "name": _names,
        "safety_score": safety_vals,
    })

    COLOR_MAP = {
        "HH (High-High)": "#2ca02c",
        "LL (Low-Low)": "#d62728",
        "HL (High-Low)": "#ff7f0e",
        "LH (Low-High)": "#9467bd",
    }

    fig_moran = px.scatter(
        scatter_df,
        x="safety_std",
        y="spatial_lag_std",
        color="quadrant",
        color_discrete_map=COLOR_MAP,
        opacity=0.6,
        hover_name="name",
        hover_data={"safety_score": ":.1f", "safety_std": False, "spatial_lag_std": False},
        title=f"Moran Scatter Plot  (I = {morans_I:.4f})",
        labels={
            "safety_std": "Standardised Safety Score",
            "spatial_lag_std": "Spatially Lagged Safety Score (standardised)",
            "quadrant": "LISA Quadrant",
        },
    )
    fig_moran.add_hline(y=0, line_color="black", line_width=0.8)
    fig_moran.add_vline(x=0, line_color="black", line_width=0.8)
    # Moran's I regression line
    slope = np.polyfit(x_std, lag_std, 1)[0]
    x_line = np.linspace(x_std.min(), x_std.max(), 100)
    fig_moran.add_trace(go.Scatter(
        x=x_line, y=slope * x_line,
        mode="lines", line=dict(color="black", dash="dash", width=1.5),
        name=f"Slope = I = {morans_I:.4f}", showlegend=True,
    ))
    fig_moran.update_traces(marker_size=5)
    fig_moran.update_layout(height=480)
    st.plotly_chart(fig_moran, use_container_width=True)

    st.caption(
        "HH = High safety surrounded by high safety (safe clusters). "
        "LL = Low safety surrounded by low safety (danger clusters). "
        "HL = Safe island in a dangerous region. "
        "LH = Danger pocket in a safe region."
    )

    # Summary tables: HH and LL counties
    col_hh, col_ll = st.columns(2)
    with col_hh:
        st.markdown("**HH: Safe clusters** (top 15 by safety score)")
        hh_df = scatter_df[scatter_df["quadrant"] == "HH (High-High)"].nlargest(15, "safety_score")
        st.dataframe(hh_df[["name", "safety_score"]].round(1).reset_index(drop=True), use_container_width=True)
    with col_ll:
        st.markdown("**LL: Danger clusters** (bottom 15 by safety score)")
        ll_df = scatter_df[scatter_df["quadrant"] == "LL (Low-Low)"].nsmallest(15, "safety_score")
        st.dataframe(ll_df[["name", "safety_score"]].round(1).reset_index(drop=True), use_container_width=True)

    with st.expander("HL / LH outliers (most interesting counties)"):
        col_hl, col_lh = st.columns(2)
        with col_hl:
            st.markdown("**HL: Safe islands** in dangerous regions")
            hl_df = scatter_df[scatter_df["quadrant"] == "HL (High-Low)"].sort_values("safety_score", ascending=False)
            st.dataframe(hl_df[["name", "safety_score"]].round(1).reset_index(drop=True), use_container_width=True)
        with col_lh:
            st.markdown("**LH: Danger pockets** in safe regions")
            lh_df = scatter_df[scatter_df["quadrant"] == "LH (Low-High)"].sort_values("safety_score")
            st.dataframe(lh_df[["name", "safety_score"]].round(1).reset_index(drop=True), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: VALIDATION (Convergent validity vs life expectancy)
# ══════════════════════════════════════════════════════════════════════════════

with tab_valid:
    st.subheader("Convergent Validity: Safety Score vs Life Expectancy")
    st.markdown(
        "A well-constructed safety index should correlate with real-world outcomes it never observed. "
        "Life expectancy (County Health Rankings 2023) was **not used as an input**; "
        "a strong positive Spearman correlation confirms the index tracks true county-level risk."
    )

    _bayes_k_val = bayes_k if traffic_method == "Bayesian Shrinkage" else 10
    vdata = compute_validation_data(tuple(weights), traffic_method, _bayes_k_val)
    merged = vdata["merged"]
    rho, p_rho = vdata["rho"], vdata["p_rho"]
    r = vdata["r"]

    if len(merged) == 0:
        st.warning("Could not merge safety scores with life expectancy data.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Counties matched", f"{len(merged):,}")
        m2.metric("Spearman ρ", f"{rho:.4f}", help="Rank-based; robust to outliers")
        m3.metric("Pearson r", f"{r:.4f}", help="Linear correlation")
        m4.metric("p-value", "< 0.001" if p_rho < 0.001 else f"{p_rho:.3e}")

        if rho > 0.7:
            st.success(f"Strong convergent validity (ρ = {rho:.3f}). Counties with higher safety scores live significantly longer on average.")
        elif rho > 0.5:
            st.info(f"Moderate convergent validity (ρ = {rho:.3f}). The index tracks real-world outcomes reasonably well.")
        else:
            st.warning(f"Weak convergent validity (ρ = {rho:.3f}). Consider revising sub-score weights.")

        # Scatter: safety_score vs life_expectancy
        fig_le = px.scatter(
            merged,
            x="safety_score",
            y="life_expectancy",
            color="census_region",
            opacity=0.55,
            hover_name="name",
            hover_data={"safety_score": ":.1f", "life_expectancy": ":.1f", "census_region": True},
            trendline="ols",
            trendline_scope="overall",
            trendline_color_override="black",
            labels={
                "safety_score": f"Safety Score (0-100, weights: {preset_name})",
                "life_expectancy": "Life Expectancy (years, CHR 2023)",
                "census_region": "Region",
            },
            title=f"Safety Score vs Life Expectancy  (Spearman ρ = {rho:.4f}, Pearson r = {r:.4f})",
        )
        fig_le.update_traces(marker_size=5)
        fig_le.update_layout(height=480)
        st.plotly_chart(fig_le, use_container_width=True)
        st.caption(
            "Trend line is OLS across all counties. Color = Census region. "
            "Life expectancy source: County Health Rankings 2023 (CHR, Robert Wood Johnson Foundation). "
            "Not used in index construction; serves as an independent benchmark."
        )

        # Sub-score correlations
        st.subheader("Sub-score Correlations with Life Expectancy")
        st.markdown("Which dimension drives the most of the predictive power?")

        sub_corr_df = pd.DataFrame(vdata["sub_corrs"]).sort_values("Spearman ρ", ascending=True)
        fig_bar = px.bar(
            sub_corr_df,
            x="Spearman ρ",
            y="Sub-score",
            orientation="h",
            color="Spearman ρ",
            color_continuous_scale="RdYlGn",
            range_color=(0, 1),
            text=sub_corr_df["Spearman ρ"].map("{:.3f}".format),
            title="Spearman ρ with Life Expectancy by Sub-score",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            height=280,
            xaxis=dict(range=[0, 1]),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Additional outcomes
        with st.expander("Additional external benchmarks"):
            if vdata["available_outcomes"]:
                st.markdown("**Composite safety score vs external outcomes**")
                st.dataframe(pd.DataFrame(vdata["bench_rows"]), use_container_width=True)
                st.caption("Negative correlations are correct direction: higher safety score → lower mortality/disease burden.")

                st.divider()

                st.markdown("**Sub-score correlations with each external outcome**")
                heatmap_df = pd.DataFrame(vdata["heatmap_data"], index=SCORE_LABELS)
                fig_heat = go.Figure(go.Heatmap(
                    z=heatmap_df.values,
                    x=heatmap_df.columns.tolist(),
                    y=heatmap_df.index.tolist(),
                    colorscale="RdYlGn",
                    zmin=-1, zmax=1,
                    text=heatmap_df.values,
                    texttemplate="%{text:.3f}",
                    showscale=True,
                    colorbar=dict(title="Spearman ρ"),
                ))
                fig_heat.update_layout(
                    title="Spearman ρ: Sub-score vs External Outcome",
                    height=320,
                    xaxis=dict(tickangle=-20),
                )
                st.plotly_chart(fig_heat, use_container_width=True)
                st.caption(
                    "Life expectancy: positive rho is correct (safer = longer life). "
                    "Mortality outcomes: negative rho is correct (safer = lower mortality)."
                )
