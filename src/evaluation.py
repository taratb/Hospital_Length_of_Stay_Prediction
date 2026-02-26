import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import(
    mean_absolute_error,
    mean_squared_error
)

def regression_metrics(y_true, y_pred):

    return {
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

def plot_residuals(y_true, y_pred, title="Analiza reziduala"):
    """
    Prikazuje dva grafika:
    - Reziduali vs predvidjene vrednosti
    - Histogram reziduala (provera normalnosti)
    """
    e = residuals(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Reziduali vs predviđene vrednosti
    axes[0].scatter(y_pred, e, alpha=0.3, s=5)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xlabel("Predviđene vrednosti")
    axes[0].set_ylabel("Reziduali")
    axes[0].set_title(f"{title} — reziduali vs predviđene vrednosti")

    # Histogram reziduala
    axes[1].hist(e, bins=50, edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--')
    axes[1].set_xlabel("Rezidual")
    axes[1].set_ylabel("Frekvencija")
    axes[1].set_title(f"{title} — distribucija reziduala")

    plt.tight_layout()
    plt.show()

    