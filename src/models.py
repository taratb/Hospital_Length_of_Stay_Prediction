import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor


# --------------------------------------------------
# Podela na train / test
# --------------------------------------------------

def split_data(df, features, target, test_size=0.2, random_state=42):
    """
    Deli podatke na trening i test skup.
    """
    X = df[features]
    y = df[target]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )


# --------------------------------------------------
# Treniranje modela
# --------------------------------------------------

def train_random_forest(X_train, y_train, random_state=42):
    """
    Treniranje Random Forest regresionog modela.
    """
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, random_state=42):
    """
    Treniranje XGBoost regresionog modela.
    """
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# --------------------------------------------------
# Evaluacija modela
# --------------------------------------------------

def evaluate_regression_model(model, X_test, y_test):
    """
    Računa MAE i RMSE metrike za regresioni model.
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {
        "MAE": mae,
        "RMSE": rmse
    }


# --------------------------------------------------
# Feature importance (jako korisno za rad)
# --------------------------------------------------

def get_feature_importance(model, feature_names):
    """
    Vraća DataFrame sa značajem atributa.
    Radi i za Random Forest i za XGBoost.
    """
    importance = model.feature_importances_

    return (
        pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )