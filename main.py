import matplotlib.pyplot as plt
from src.load_data import load_data
from src.eda import *
from src.preprocessing import *
from src.clustering import *

def main():
    RUN_PLOTS = False

    df = load_data('data/LengthOfStay.csv')

    # print(df.shape)
    # print(df.head())
    # print(df.columns)
    # print(df.dtypes)

    #print(df['lengthofstay'].describe())
    print(f"Mean: {df['lengthofstay'].mean()}")
    print(f"Medijana: {df['lengthofstay'].median()}")

    if RUN_PLOTS:
        plot_target_distribution(df['lengthofstay'])
        plot_target_outliers(df['lengthofstay'])

#______________________________________________________________________________________
# pretpocesiranje podataka
#______________________________________________________________________________________

    #provera nedostajucih vrednosti
    print(missing_values(df))

    df = drop_non_informative_columns(df)
    df = encode_categorical_features(df)
    df = encode_boolean_features(df)
    print(df.head(5).T)

#______________________________________________________________________________________
# analiza podataka
#______________________________________________________________________________________

    # matrica korelacije
    correlation_with_target(df)
    if RUN_PLOTS:
        plot_correlation_heatmap(df)

    # podela po grupama
    binary_cols = ['gender_M','dialysisrenalendstage','malnutrition','asthma','hemo']
    for col in binary_cols:
        if RUN_PLOTS: plot_los_by_category(df, col)

    # rcount
    if RUN_PLOTS:
        plot_median_los_by_rcount(df)

    # poredjenje pacijenata sa i bez odredjenih karakteristika
    conditions = [
        "asthma","pneum","depress","malnutrition","hemo",
        "dialysisrenalendstage","psychologicaldisordermajor","psychother",
        "irondef","fibrosisandother","substancedependence"
    ]
    effects_df = compare_binary_features(
        df,
        binary_cols=conditions,
        target="lengthofstay",
        agg="median"
    )

    print("\nTop binarni efekti (medijana):")
    print(effects_df.head(11))

    if RUN_PLOTS:
        plot_top_binary_effects(
            df,
            binary_cols=conditions,
            target="lengthofstay",
            top_n=8,
            agg="median",
            min_n1=50
        )


#______________________________________________________________________________________
# podela na klastere
#______________________________________________________________________________________


    cluster_features = [
        'creatinine',
        'bloodureanitro',
        'glucose',
        'sodium',
        'hematocrit',
        'bmi',
        'pulse',
        'respiration',
        'lengthofstay'
    ]

    X = prepare_clustering_data(df, cluster_features)
    X_scaled, _ = scale_features(X)

    elbow_df = compute_elbow(X_scaled, k_min=2, k_max=8)
    print(elbow_df)

    labels, _ = fit_kmeans(X_scaled, n_clusters=3)

    summary = cluster_summary(X, labels)
    print(summary)

    plot_clusters_pca(X_scaled, labels)
if __name__ == '__main__':
    main()