import numpy as np
import pandas as pd


def save_complex_vector(filename, x):
    with open(filename, "w") as f:
        # write dimensions
        f.write(f"{x.shape[0]} \n")
        # write real and imag parts side by side
        for complex in x:
            f.write(f"  ({complex.real:.6f},{complex.imag:.6f})\n")
        f.write("\n")


def export_all_metrics_to_csv(algo, filename):
    data = {
        "f": np.array(algo.f_values),
        "mse": np.array(algo.mse_values),
        "mae": np.array(algo.mae_values),
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
