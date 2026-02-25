import numpy as np

from sklearn.metrics import(
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def regression_metrics(y_true, y_pred):

    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred)))
    }

def evaluate_model(model, x, y):
    y_pred = model.predict(x)
    return regression_metrics(y, y_pred)

def residuals(y_true, y_pred):
    return y_true - y_pred

def residual_summary(y_true, y_pred):
    """
    Proverava da li ima smisla koristiti linearnu regresiju
    """
    e = residuals(y_true, y_pred)

    return {
        "residual_mean": float(np.mean(e)),
        "residual_median": float(np.median(e)),
        "residual_std": float(np.std(e)),
        "residual_min": float(np.min(e)),
        "residual_max": float(np.max(e)),
    }