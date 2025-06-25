import matplotlib.pyplot as plt
import numpy as np
from tueplots.constants.color import rgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas import DataFrame


def plot_most_influential_chained_prediction(importances, n_base_features, target_names=None):
    """
    Plot the index of the most influential previous prediction for each regressor in the chain.
    
    Parameters:
    - importances: list of 1D arrays, one per regressor.
    - n_base_features: int, number of original (non-chained) features.
    - target_names: optional list of target names, for labeling bars.
    """
    n_targets = len(importances)
    
    influential_preds = []
    for i, imp in enumerate(importances):
        chain_features = imp[n_base_features:]
        if len(chain_features) == 0:
            influential_preds.append(None)
        else:
            influential_preds.append(np.argmax(chain_features))
    
    x_labels = [f"{i}" for i in range(n_targets)]
    y_labels = [f"{i}" if i is not None else "None" for i in influential_preds]
    
    y_values = [
        p if p is not None else -1 for p in influential_preds
    ]
    
    plt.figure(figsize=(10, 4))
    bars = plt.bar(range(n_targets), y_values, tick_label=x_labels, color=rgb.tue_blue)

    for idx, bar in enumerate(bars):
        height = bar.get_height()
        label = y_labels[idx]
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.1, label,
                 ha='center', va='bottom')
    
    plt.title("Most Influential Previous Prediction per Regressor")
    plt.xlabel("target time step t")
    plt.ylabel("Index of Most Influential Previous Prediction")
    plt.ylim(-1, max(filter(None, y_values), default=0) + 1.5)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()
    
def plot_error_over_horizon(y_test_24: DataFrame, pred_chain: np.ndarray, horizon: int):
    # Plot test set errors over the horizon
    mae_values = []
    rmse_values = []

    for i in range(horizon):
        mae = mean_absolute_error(y_test_24.iloc[:, i], pred_chain[:, i])
        rmse = np.sqrt(mean_squared_error(y_test_24.iloc[:, i], pred_chain[:, i]))
        mae_values.append(mae)
        rmse_values.append(rmse)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, horizon + 1), mae_values, label="MAE", marker='o', color=rgb.tue_blue)
    plt.plot(range(1, horizon + 1), rmse_values, label="RMSE", marker='s', color=rgb.tue_red)
    plt.xlabel("Horizon (hours)")
    plt.ylabel("Error")
    plt.title("Individual Test Set Errors Over the Horizon")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    