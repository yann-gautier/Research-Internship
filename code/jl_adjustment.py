import numpy as np
import pandas as pd
from sktime.datasets import load_from_tsfile_to_dataframe

def get_jl_output_dim_from_data(X, epsilon: float) -> int:
    if not (0 < epsilon < 1):
        raise ValueError("epsilon doit être entre 0 et 1")

    if isinstance(X, pd.DataFrame):
        n = X.shape[0]
    elif isinstance(X, np.ndarray):
        n = X.shape[0]
    else:
        raise TypeError("X doit être un DataFrame ou un ndarray")

    return int(np.ceil(4 * np.log(n) / (epsilon ** 2)))

def get_jl_output_dim_from_tsfile(filepath: str, epsilon: float) -> int:
    X, _ = load_from_tsfile_to_dataframe(filepath)
    return get_jl_output_dim_from_data(X, epsilon)