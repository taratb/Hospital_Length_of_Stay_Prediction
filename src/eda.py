import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def basic_stats(series):
    return series.describe()

# ciljna promenljiva
def plot_target_distribution(target):
    plt.figure(figsize=(7, 4))
    plt.hist(target, bins=50)
    plt.xlabel("Length of stay (days)")
    plt.ylabel("Frequency")
    plt.title("Distribucija ciljne promenljive lengthofstay")
    plt.show()

def plot_target_outliers(target):
    plt.figure(figsize=(6, 2))
    plt.boxplot(target, vert=False)
    plt.xlabel("Length of stay (days)")
    plt.title("Boxplot ciljne promenljive lengthofstay")
    plt.show()

# provera nedostajucih vrednosti
def missing_values(df):
    missing_percent = (df.isnull().sum() )
    return missing_percent


# korelacije
def correlation_with_target(df, target = "lengthofstay"):
    corr = df.corr(numeric_only=True)[target].sort_values(ascending=False)
    return corr

def plot_correlation_heatmap(df, cols=None, title="Korelaciona matrica"):
    data = df[cols] if cols is not None else df
    corr = data.corr(numeric_only=True)

    plt.figure(figsize=(9, 7))
    plt.imshow(corr, cmap="coolwarm")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# odnos izmedju ciljne promenljive i zadate
def plot_los_by_category(df, category_col):
    #plt.figure(figsize=(6, 5))
    df.boxplot(column='lengthofstay', by=category_col)
    plt.title(f'Length of Stay vs {category_col}')
    plt.suptitle('')
    plt.xlabel(category_col)
    plt.ylabel('Length of Stay')
    plt.ylim(0, 17)
    plt.show()


# Prikazuje kako se medijana trajanja hospitalizacije menja
# u zavisnosti od broja prethodnih prijema pacijenata.
def plot_median_los_by_rcount(df):
    rcount_cols = ['rcount_1', 'rcount_2', 'rcount_3', 'rcount_4', 'rcount_5+']
    labels = ['0', '1', '2', '3', '4', '5+']

    # rcount=0: nijedna one-hot nije 1
    mask_zero = (df[rcount_cols].sum(axis=1) == 0)

    medians = []
    s0 = df.loc[mask_zero, 'lengthofstay'].dropna()
    medians.append(np.nan if s0.empty else s0.median())

    for col in rcount_cols:
        s = df.loc[df[col] == 1, 'lengthofstay'].dropna()
        medians.append(np.nan if s.empty else s.median())

    plt.figure(figsize=(7, 5))
    plt.plot(labels, medians, marker='o')
    plt.xlabel('Broj prethodnih prijema (rcount)')
    plt.ylabel('Medijana trajanja hospitalizacije (dani)')
    plt.title('Medijana trajanja hospitalizacije po rcount kategorijama')
    plt.ylim(0, 17)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def compare_binary_features(df, binary_cols, target="lengthofstay", agg="median"):
    """
    Za svaku binarnu kolonu računa:
    - agg(target | 0), agg(target | 1)
    - diff = agg1 - agg0
    - n0, n1 (broj uzoraka)
    """
    rows = []

    for col in binary_cols:
        s = df[[col, target]].dropna()

        if agg == "mean":
            grp = s.groupby(col)[target].mean()
        else:
            grp = s.groupby(col)[target].median()

        counts = s.groupby(col)[target].size()

        v0 = grp.get(0, np.nan)
        v1 = grp.get(1, np.nan)
        n0 = int(counts.get(0, 0))
        n1 = int(counts.get(1, 0))

        rows.append({
            "feature": col,
            f"{agg}_0": float(v0) if not pd.isna(v0) else np.nan,
            f"{agg}_1": float(v1) if not pd.isna(v1) else np.nan,
            "diff_1_minus_0": (float(v1) - float(v0))
                               if (not pd.isna(v0) and not pd.isna(v1)) else np.nan,
            "n0": n0,
            "n1": n1
        })

    out = pd.DataFrame(rows).sort_values("diff_1_minus_0", ascending=False)
    return out


def plot_top_binary_effects(df, binary_cols, target="lengthofstay",
                            top_n=10, agg="median", min_n1=50):
    """
    Prikazuje top N binarnih faktora po razlici agg(target|1)-agg(target|0),
    uz filter da u grupi '1' ima bar min_n1 uzoraka.
    """
    effects = compare_binary_features(df, binary_cols, target=target, agg=agg)
    effects = effects.dropna(subset=["diff_1_minus_0"])
    effects = effects[effects["n1"] >= min_n1].head(top_n)

    plt.figure(figsize=(10, 4))
    plt.bar(effects["feature"], effects["diff_1_minus_0"].values)
    plt.xticks(rotation=90)
    plt.ylabel(f"{agg}(1) - {agg}(0)")
    plt.title(f"Top efekti binarnih faktora ({agg} razlika), min n1={min_n1}")
    plt.tight_layout()
    plt.show()

    return effects
