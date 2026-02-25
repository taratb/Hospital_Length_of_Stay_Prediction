import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --------------------------------------------------
# Priprema podataka za klasterizaciju
# --------------------------------------------------

def prepare_clustering_data(df, features):
    """
    Izvlači odabrane numeričke kolone i uklanja redove sa NaN vrednostima.
    """
    X = df[features].copy()
    X = X.dropna()
    return X


def scale_features(X):
    """
    Standardizuje podatke (mean=0, std=1) – obavezno za K-Means.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# --------------------------------------------------
# Elbow metoda
# --------------------------------------------------

def compute_elbow(X_scaled, k_min=2, k_max=8, random_state=42):
    """
    Računa inertia vrednosti za Elbow metodu.
    Vraća DataFrame sa kolonama: k, inertia.
    """
    inertias = []

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10
        )
        kmeans.fit(X_scaled)
        inertias.append({
            "k": k,
            "inertia": kmeans.inertia_
        })

    return pd.DataFrame(inertias)


# --------------------------------------------------
# Treniranje K-Means modela
# --------------------------------------------------

def fit_kmeans(X_scaled, n_clusters, random_state=42):
    """
    Treniranje K-Means modela i dodela klaster labela.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    labels = kmeans.fit_predict(X_scaled)
    return labels, kmeans


# --------------------------------------------------
# Analiza klastera
# --------------------------------------------------

def cluster_summary(X, labels):
    """
    Računa srednje vrednosti i veličinu svakog klastera.
    """
    df_tmp = X.copy()
    df_tmp["cluster"] = labels

    summary = (
        df_tmp
        .groupby("cluster")
        .agg(["mean", "std", "count"])
        .round(2)
    )

    return summary


def plot_clusters_pca(X_scaled, labels):
    """
    Prikazuje klastere u 2D prostoru korišćenjem PCA transformacije.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(6, 5))
    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        alpha=0.5
    )
    plt.xlabel('PCA komponenta 1')
    plt.ylabel('PCA komponenta 2')
    plt.title('Vizualizacija klastera (PCA)')
    plt.tight_layout()
    plt.show()

    return pca.explained_variance_ratio_