import matplotlib.pyplot as plt
import pandas as pd
from src.eda import *
from src.models import *
from src.evaluation import *
from src.preprocessing import *
from src.clustering import *

def load_data(path):
    df = pd.read_csv(path)
    return df

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
    #print(missing_values(df))

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
        'creatinine', 'bloodureanitro', 'glucose', 'sodium',
        'hematocrit', 'bmi', 'pulse', 'respiration', 'lengthofstay'
    ]

    _, _, summary = run_clustering(df, cluster_features, n_clusters=3)
    print(summary)

    

#______________________________________________________________________________________
# modeli
#______________________________________________________________________________________

    features = [col for col in df.columns if col != 'lengthofstay']

    x_train, x_val, x_test, y_train, y_val, y_test = split_data_train_val_test(
        df, features, target='lengthofstay'
    )

    # Linearna regresija
    lr_model = train_linear_regression(x_train, y_train)
    print("\nLinearna regresija — val skup:")
    print(evaluate_model(lr_model, x_val, y_val))
    print("Reziduali (val):")
    print(residual_summary(y_val, lr_model.predict(x_val)))

    # Random Forest
    rf_model = train_random_forest(x_train, y_train)
    print("\nRandom Forest — val skup:")
    print(evaluate_model(rf_model, x_val, y_val))

    # XGBoost
    xgb_model = train_xgboost(x_train, y_train, x_val, y_val)
    print("\nXGBoost — val skup:")
    print(evaluate_model(xgb_model, x_val, y_val))

    # Finalna evaluacija na test skupu
    print("\n=== FINALNA EVALUACIJA (test skup) ===")
    print("Linearna regresija:", evaluate_model(lr_model, x_test, y_test))
    print("Random Forest:     ", evaluate_model(rf_model, x_test, y_test))
    print("XGBoost:           ", evaluate_model(xgb_model, x_test, y_test))

if __name__ == '__main__':
    main()