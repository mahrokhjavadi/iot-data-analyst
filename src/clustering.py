# =============================================================================
# CLASS 03 – ADVANCED CLUSTERING ANALYSIS (FINAL VERSION)
# =============================================================================
# - Loads the same cleaned CSV as corelation_new.py
# - Uses:
#     * Elbow Method + Silhouette Score
#     * K-Means++ (main baseline)
#     * Hierarchical Clustering + Dendrogram
#     * DBSCAN (density-based)
#     * Gaussian Mixture Model (GMM) – modern probabilistic clustering
# - Saves:
#     * 9_elbow_method.png
#     * 10_dendrogram.png
#     * 11_clustering_visualization.png
#     * kmeans_cluster_statistics.csv
#     * clustering_results.csv
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage

plt.style.use("ggplot")

print("\n" + "=" * 80)
print("تحلیل خوشه‌بندی پیشرفته داده‌ها – CLASS 03")
print("=" * 80)

# =============================================================================
# 1. CONFIGURATION & LOAD DATA
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(
    BASE_DIR,
    "cleaned_datasets",
    "cleaned_data_20251126_195100.csv"
)

df_cleaned = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"\n✓ Loaded {len(df_cleaned):,} rows from: {INPUT_FILE}")

# =============================================================================
# 2. FEATURE SELECTION FOR CLUSTERING
# =============================================================================

# IMPORTANT: column names are aligned with corelation_new.py
# (Voltage (V), Current (A), Power Consumption (kW), Power Factor, Reactive Power (kVAR))
clustering_features = [
    "Voltage (V)",
    "Current (A)",
    "Power Consumption (kW)",
    "Power Factor",
    "Reactive Power (kVAR)",
]

# Keep only columns that actually exist in the CSV
clustering_features = [col for col in clustering_features if col in df_cleaned.columns]

if len(clustering_features) < 2:
    print("\n❌ تعداد ویژگی‌های کافی برای خوشه‌بندی موجود نیست!")
    print("   Features found:", clustering_features)
else:
    print("\n✓ Features used for clustering:")
    for c in clustering_features:
        print("  -", c)

    # Drop rows with NaNs in selected features
    X = df_cleaned[clustering_features].dropna()
    print(f"\n✓ Number of samples used for clustering: {len(X):,}")

    # Keep original index to be able to merge results back later
    original_index = X.index

    # =============================================================================
    # 3. SCALING
    # =============================================================================

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =============================================================================
    # 4. FIND OPTIMAL K (ELBOW + SILHOUETTE)
    # =============================================================================

    print("\n" + "-" * 80)
    print("1) یافتن تعداد بهینه خوشه‌ها (Elbow + Silhouette)")
    print("-" * 80)

    inertias = []
    silhouette_scores_list = []
    K_range = range(2, 11)

    for k in K_range:
        kmeans = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=15,
            random_state=42
        )
        kmeans.fit(X_scaled)
        labels = kmeans.labels_
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(X_scaled, labels)
        silhouette_scores_list.append(sil_score)

        print(f"  k={k:2d} → Inertia={kmeans.inertia_:,.0f}, Silhouette={sil_score:.4f}")

    # Plot Elbow + Silhouette
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(list(K_range), inertias, "bo-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Number of Clusters (k)", fontsize=12)
    axes[0].set_ylabel("Inertia (Within-cluster SS)", fontsize=12)
    axes[0].set_title("Elbow Method – Inertia", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(K_range), silhouette_scores_list, "ro-", linewidth=2, markersize=8)
    axes[1].set_xlabel("Number of Clusters (k)", fontsize=12)
    axes[1].set_ylabel("Silhouette Score", fontsize=12)
    axes[1].set_title("Silhouette Score vs k", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    elbow_file = os.path.join(BASE_DIR, "9_elbow_method.png")
    plt.savefig(elbow_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Elbow & Silhouette plot saved → {elbow_file}")

    # Choose best k by max silhouette
    optimal_k = list(K_range)[silhouette_scores_list.index(max(silhouette_scores_list))]
    print(f"\n→ تعداد بهینه خوشه‌ها بر اساس Silhouette Score: k = {optimal_k}")

    # =============================================================================
    # 5. K-MEANS++ CLUSTERING (MAIN BASELINE)
    # =============================================================================

    print("\n" + "-" * 80)
    print(f"2) K-Means++ Clustering (k={optimal_k})")
    print("-" * 80)

    kmeans = KMeans(
        n_clusters=optimal_k,
        init="k-means++",
        n_init=20,
        random_state=42
    )
    kmeans_labels = kmeans.fit_predict(X_scaled)

    sil_km = silhouette_score(X_scaled, kmeans_labels)
    db_km = davies_bouldin_score(X_scaled, kmeans_labels)
    ch_km = calinski_harabasz_score(X_scaled, kmeans_labels)

    print(f"   Silhouette Score:       {sil_km:.4f} (هرچه نزدیک‌تر به 1 بهتر)")
    print(f"   Davies-Bouldin Index:   {db_km:.4f} (هرچه کوچک‌تر بهتر)")
    print(f"   Calinski-Harabasz Index:{ch_km:.4f} (هرچه بزرگ‌تر بهتر)")

    # Create result DataFrame aligned with original sample index
    df_clusters = X.copy()
    df_clusters["Cluster_KMeans"] = kmeans_labels

    # Per-cluster statistics
    print("\n   آمار خوشه‌های K-Means:")
    cluster_stats = df_clusters.groupby("Cluster_KMeans").agg(["mean", "count"])
    print(cluster_stats)

    stats_file = os.path.join(BASE_DIR, "kmeans_cluster_statistics.csv")
    cluster_stats.to_csv(stats_file, encoding="utf-8-sig")
    print(f"\n✓ KMeans cluster statistics saved → {stats_file}")

    # =============================================================================
    # 6. HIERARCHICAL CLUSTERING + DENDROGRAM
    # =============================================================================

    print("\n" + "-" * 80)
    print("3) خوشه‌بندی سلسله‌مراتبی (Hierarchical)")
    print("-" * 80)

    # Sample subset for dendrogram (for speed)
    sample_size = min(1000, len(X_scaled))
    sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
    X_sample = X_scaled[sample_indices]

    linkage_matrix = linkage(X_sample, method="ward")

    plt.figure(figsize=(16, 8))
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90,
        leaf_font_size=10,
        show_contracted=True,
    )
    plt.title("Hierarchical Clustering Dendrogram", fontsize=16, fontweight="bold")
    plt.xlabel("Sample Index or (Cluster Size)", fontsize=12)
    plt.ylabel("Distance", fontsize=12)
    plt.tight_layout()
    dendro_file = os.path.join(BASE_DIR, "10_dendrogram.png")
    plt.savefig(dendro_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"   ✓ Dendrogram saved → {dendro_file}")

    # Agglomerative clustering for all points
    hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
    hierarchical_labels = hierarchical.fit_predict(X_scaled)
    df_clusters["Cluster_Hierarchical"] = hierarchical_labels

    # =============================================================================
    # 7. DBSCAN (DENSITY-BASED)
    # =============================================================================

    print("\n" + "-" * 80)
    print("4) DBSCAN Clustering (Density-based)")
    print("-" * 80)

    # Use k-distance plot idea to estimate eps (automated via percentile)
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)
    distances = np.sort(distances[:, -1], axis=0)

    # Use 95th percentile as a heuristic for eps
    eps = np.percentile(distances, 95)

    dbscan = DBSCAN(eps=eps, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)

    print(f"   Estimated eps:          {eps:.4f}")
    print(f"   Number of clusters:     {n_clusters_db}")
    print(f"   Number of noise points: {n_noise} ({(n_noise/len(X_scaled))*100:.2f}%)")

    df_clusters["Cluster_DBSCAN"] = dbscan_labels

    # =============================================================================
    # 8. GAUSSIAN MIXTURE MODEL (GMM) – MODERN METHOD
    # =============================================================================

    print("\n" + "-" * 80)
    print("5) Gaussian Mixture Model (GMM)")
    print("-" * 80)

    gmm = GaussianMixture(
        n_components=optimal_k,
        covariance_type="full",
        random_state=42
    )
    gmm_labels = gmm.fit_predict(X_scaled)

    sil_gmm = silhouette_score(X_scaled, gmm_labels)
    print(f"   Silhouette Score (GMM): {sil_gmm:.4f}")

    df_clusters["Cluster_GMM"] = gmm_labels

    # =============================================================================
    # 9. PCA-BASED VISUALIZATION OF CLUSTERS
    # =============================================================================

    print("\n" + "-" * 80)
    print("6) تجسم خوشه‌ها با کاهش بُعد (PCA)")
    print("-" * 80)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"   واریانس توضیح داده شده توسط 2 مؤلفه اول: {var_explained:.2f}%")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # KMeans
    sc1 = axes[0, 0].scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=kmeans_labels, cmap="viridis", s=10, alpha=0.7
    )
    axes[0, 0].set_title(f"K-Means++ (k={optimal_k})")
    axes[0, 0].set_xlabel("PC1")
    axes[0, 0].set_ylabel("PC2")
    plt.colorbar(sc1, ax=axes[0, 0], label="Cluster")

    # Hierarchical
    sc2 = axes[0, 1].scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=hierarchical_labels, cmap="plasma", s=10, alpha=0.7
    )
    axes[0, 1].set_title("Hierarchical Clustering")
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")
    plt.colorbar(sc2, ax=axes[0, 1], label="Cluster")

    # DBSCAN
    sc3 = axes[1, 0].scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=dbscan_labels, cmap="Spectral", s=10, alpha=0.7
    )
    axes[1, 0].set_title(f"DBSCAN (eps={eps:.3f})")
    axes[1, 0].set_xlabel("PC1")
    axes[1, 0].set_ylabel("PC2")
    plt.colorbar(sc3, ax=axes[1, 0], label="Cluster (-1 = Noise)")

    # GMM
    sc4 = axes[1, 1].scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=gmm_labels, cmap="coolwarm", s=10, alpha=0.7
    )
    axes[1, 1].set_title("Gaussian Mixture Model (GMM)")
    axes[1, 1].set_xlabel("PC1")
    axes[1, 1].set_ylabel("PC2")
    plt.colorbar(sc4, ax=axes[1, 1], label="Cluster")

    plt.tight_layout()
    cluster_vis_file = os.path.join(BASE_DIR, "11_clustering_visualization.png")
    plt.savefig(cluster_vis_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"   ✓ Cluster visualization saved → {cluster_vis_file}")

    # =============================================================================
    # 10. SAVE CLUSTERING RESULTS
    # =============================================================================

    # align with original df_cleaned by index
    result_df = df_cleaned.copy()
    for col in df_clusters.columns:
        if col.startswith("Cluster_"):
            # fill with NaN then put labels at original indices
            result_df[col] = np.nan
            result_df.loc[original_index, col] = df_clusters[col].values

    output_file = os.path.join(BASE_DIR, "clustering_results.csv")
    result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n✓ نتایج خوشه‌بندی در فایل ذخیره شد → {output_file}")

print("\n" + "=" * 80)
print("ADVANCED CLUSTERING ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 80)
