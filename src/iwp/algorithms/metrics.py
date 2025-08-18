import numpy as np


def mae(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE) between true and predicted values.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.

    Returns:
    float: The mean absolute error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("Shape of y_true and y_pred must match.")

    return np.mean(np.abs(y_true - y_pred))


def mse(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.

    Returns:
    float: The mean squared error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("Shape of y_true and y_pred must match.")

    return np.mean((y_true - y_pred) ** 2)
