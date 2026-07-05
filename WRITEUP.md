# Urban Safety Atlas: County-Level Risk Index for 3,144 US Counties

## Overview

The Urban Safety Atlas constructs a composite safety index for every US county by merging five government data sources, quantifies how safety clusters geographically, validates the index against an external outcome never used in its construction, and exposes all results through an interactive Streamlit dashboard. The project covers the full data science pipeline: acquisition, imputation, feature engineering, clustering, statistical testing, and external validation.

---

## Data Sources

| Source | Agency | Variable captured |
|---|---|---|
| CDC PLACES 2023 | CDC | Health outcomes (chronic disease, uninsured rate, poor health days) |
| AQI Summary 2023 | EPA | Air quality (annual AQI, days unhealthy) |
| FARS 2023 | NHTSA | Traffic fatalities per 100K residents |
| SAIPE 2023 | US Census | Economic distress (poverty rate, median household income) |
| County Health Rankings 2023 | Robert Wood Johnson Foundation | Crime (homicide rate, firearm fatality rate per 100K); life expectancy used for validation only |

---

## Feature Engineering

Each data source contributes one sub-score on a 0 to 100 scale (higher = safer):

- **Health score**: inverse-scaled from poor health outcomes and uninsured rate
- **Air quality score**: inverse-scaled from median AQI and unhealthy air days; 2,170 counties (69%) lack EPA monitors and received imputed values via Inverse Distance Weighting (IDW) from the nearest monitored counties
- **Economic score**: scaled from poverty rate and median household income
- **Traffic score**: Bayesian-shrunk fatality rate, log-compressed, inverse-scaled. The raw county fatality rate is first shrunk toward the national mean using empirical Bayes smoothing. This prevents zero-fatality counties from being scored as perfectly safe. The shrunk rate is then log1p-transformed before MinMaxScaling to prevent extreme low-population outliers (e.g. Loving County, TX: population 57, rate 5,263 per 100k) from collapsing the entire scale onto a single point.
- **Crime score**: inverse-scaled from homicide rate (60% weight) and firearm fatality rate (40% weight), both per 100K residents from CHR 2023. Homicide rate receives higher weight as it measures confirmed interpersonal violence directly; firearm fatality rate captures a broader but noisier signal including suicides and accidents.

The composite **safety score** is a weighted average of the five sub-scores (default: equal weights, 0.20 each). The Streamlit dashboard allows users to adjust weights interactively. A sidebar toggle also allows switching the traffic score to a plain log variant for sensitivity comparison.

---

## Imputation

**Air quality:** EPA monitors are concentrated in urban areas. Rather than discarding rural counties, IDW imputation was applied: each unmonitored county's estimate is a distance-weighted average of monitored counties, with weight proportional to $\frac{1}{distance^2}$. This preserves all 3,144 counties in the final dataset.

**Crime:** CHR 2023 homicide rates have 42% missing coverage, reflecting counties where case counts are too small for reliable estimation. Missing values are imputed with the state median, with national median as a fallback for states where all counties are missing. Firearm fatality rates have 28% missing coverage and follow the same imputation procedure.

---

## Clustering

Two clustering methods were applied to the composite safety score:

**KMeans (k=2, silhouette score 0.283)** identified two risk tiers of roughly equal size (1,558 and 1,586 counties). The silhouette score was maximized across k = 2 to 10; k=3 scores marginally higher (0.287) but produces an unbalanced 378/1,382/1,384 split, so k=2 was retained for interpretability. Note: an earlier version of the pipeline (Bayesian shrinkage without log compression) produced an artificially high silhouette of 0.958 at k=2, caused by Loving County, TX (population 57, fatality rate 5,263 per 100k) being isolated as its own cluster. Adding log1p compression after shrinkage resolves this and produces a balanced, meaningful split.

**HDBSCAN** provided a density-based alternative that does not require specifying k in advance. Counties that do not belong to any cluster are labelled as noise (cluster = -1), which often corresponds to genuinely atypical counties.

Both methods are available in the dashboard with interactive parameter controls.

---

## Statistical Tests

### Normality: Shapiro-Wilk

Safety scores within each census region are non-normally distributed (all regions: p < 0.0001), justifying the use of non-parametric tests downstream.

### Regional Disparities: Kruskal-Wallis

The Kruskal-Wallis test confirms significant differences in safety score distributions across the four census regions (H = 805.7, p < 0.0001). Post-hoc medians:

| Region | Median safety score |
|---|---|
| South | 53.4 |
| Midwest | 58.8 |
| West | 60.8 |
| Northeast | 61.1 |

### Spatial Autocorrelation: Global Moran's I

Global Moran's I = 0.627 (p < 0.001, k = 8 nearest county centroids, row-standardized weights), indicating strong positive spatial autocorrelation. Safe counties cluster near other safe counties; unsafe counties cluster near other unsafe counties.

LISA (Local Indicators of Spatial Association) quadrant breakdown:

| Quadrant | Description | Count |
|---|---|---|
| HH | Safe county surrounded by safe neighbours | 1,288 |
| LL | Unsafe county surrounded by unsafe neighbours | 1,264 |
| HL | Safe county in an otherwise unsafe region | 281 |
| LH | Unsafe county in an otherwise safe region | 311 |

HH and LL counties represent geographic safety clusters. HL and LH counties are the most analytically interesting: they deviate sharply from their spatial context and warrant closer examination.

---

## External Validation

A well-constructed safety index should correlate with real-world outcomes it never observed. Life expectancy from County Health Rankings 2023 was withheld entirely from index construction and used only at this validation stage.

**Main benchmark:**
Spearman rho = 0.80, Pearson r = 0.74 (both p < 0.001) between safety score and life expectancy across 3,061 counties.

Two counties (Aleutians East Borough, AK and Mono County, CA) were excluded: their CHR-reported life expectancy exceeded 100 years, a biologically implausible county average arising from very small populations and wide confidence intervals (one CI spans 68 to 157 years).

**Secondary benchmarks** (directional check: higher safety should predict lower mortality):

| Outcome | Spearman rho | Result |
|---|---|---|
| Premature mortality | -0.70 | Pass |
| Child mortality | -0.61 | Pass |
| Drug overdose deaths | -0.22 | Pass |

All four external outcomes pass the directional test.

**Sub-score contributions to life expectancy:**

| Sub-score | Spearman rho with life expectancy |
|---|---|
| Health | +0.75 |
| Economic | +0.73 |
| Crime | +0.59 |
| Traffic | +0.38 |
| Air quality | +0.16 |

Health and economic sub-scores carry the most predictive power. Crime ranks third, reflecting that counties with high homicide and firearm fatality rates also tend to have shorter life expectancy. Air quality is the weakest, partly because 69% of counties received IDW-imputed rather than directly measured values. The traffic sub-score correlation reflects the Bayesian-shrunk variant; substituting the plain log variant reduces the composite Spearman rho to 0.73, confirming that shrinkage removes small-sample noise rather than genuine signal.

A sensitivity check on weights found that reducing the AQI weight to 5% (redistributing evenly to the remaining four sub-scores) nudges the composite Spearman rho from 0.80 to 0.81. The optimum across a grid search is AQI weight 8%, reaching rho = 0.811. Removing AQI entirely gives rho = 0.806. The gains are modest but consistent: any reduction in AQI weight improves the composite correlation. The equal-weight default is retained in the dashboard, but this result reinforces the data quality caveat: IDW imputation for 69% of counties flattens spatial variation in the AQI sub-score, and a lower weight better reflects its effective information content relative to the directly measured sub-scores.

Breaking the correlation down across all external outcomes reveals two additional findings. First, traffic has an unusually strong correlation with child mortality relative to its correlation with life expectancy and premature mortality. This reflects the fact that unintentional injury is the leading cause of death for children in the US, making the traffic fatality rate a near-direct input into child mortality rather than a proxy. Second, the economic sub-score outperforms the health sub-score for child mortality. The health sub-score is constructed from CDC PLACES adult chronic disease indicators (smoking, obesity, depression, poor mental health days), which have little bearing on why children die. Child mortality is driven primarily by poverty-related conditions: inadequate prenatal care, food insecurity, lack of pediatric access, and housing instability. These are captured by poverty rate and median household income. This pattern exposes a limitation of the health sub-score: it is effectively an adult health behavior index rather than a general population health measure.

---

## Dashboard

The Streamlit dashboard (`app.py`) exposes all results interactively:

- **Map tab**: choropleth of safety scores by county, filterable by census region
- **EDA tab**: score distributions, regional box plots, sub-score histograms
- **Clusters tab**: KMeans and HDBSCAN cluster maps with interactive parameter sliders (k for KMeans; min_cluster_size and min_samples for HDBSCAN)
- **County Profile tab**: lookup any county for its full score breakdown including raw homicide and firearm fatality rates
- **Spatial tab**: Moran scatter plot with LISA quadrant coloring, HH/LL/HL/LH county tables
- **Validation tab**: scatter plot of safety score vs life expectancy, sub-score correlation bar chart, secondary benchmark table

A sidebar provides preset weight schemes (Equal, Health-Heavy, Traffic-Heavy, Economic-Heavy, Crime-Heavy) and a custom slider mode that normalizes weights automatically. The sidebar also includes a traffic score method toggle: Bayesian Shrinkage (default, k=10) or Log, allowing sensitivity comparison without changing the primary results.

---

## Reproducibility

Scripts in `scripts/` can be run sequentially to reproduce all results from the raw data files.

Notebooks in `notebooks/` mirror the scripts with some outputs and commentary.

**Key dependencies:** Python 3, Pandas, NumPy, SciPy, Scikit-learn, Streamlit, Plotly

---

## Key Results at a Glance

| Finding | Value |
|---|---|
| Counties in index | 3,144 |
| Data sources | 5 |
| Counties lacking EPA monitors (IDW-imputed) | 2,170 (69%) |
| Counties with imputed homicide rates | 1,816 (58%) |
| KMeans silhouette score (k=2) | 0.283 |
| Global Moran's I | 0.627 (p < 0.001) |
| Spearman rho vs life expectancy | 0.80 (p < 0.001) |
| Regional disparity (Kruskal-Wallis) | H = 805.7 (p < 0.001) |
| South vs Northeast median safety score | 53.4 vs 61.1 |
