import numpy as np
import pandas as pd
from sktime.datasets import load_from_tsfile_to_dataframe

def get_jl_output_dim_from_data(X, epsilon: float) -> int:
    """
    Compute the target dimensionality for a Johnson-Lindenstrauss backed random projection
    based on the size of a given dataset.

    The JL lemma states that high-dimensional data can be embedded into a lower-dimensional
    space while approximately preserving pairwise distances. This function calculates the
    minimum number of dimensions required for a given distortion tolerance 'epsilon'.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The dataset for which to determine the JL projection dimensionality.
        - If 'pd.DataFrame', rows correspond to samples.
        - If 'np.ndarray', shape should be (n_samples, n_features).
    
    epsilon : float
        Distortion tolerance (0 < epsilon < 1). Smaller values preserve distances more
        accurately but require more dimensions.

    Returns
    -------
    int
        The required projection dimensionality according to the JL lemma.

    Raises
    ------
    ValueError
        If 'epsilon' is not strictly between 0 and 1.
    TypeError
        If 'X' is neither a pandas DataFrame nor a NumPy array.

    Notes
    -----
    The formula used is:
        k = ceil(4 * log(n) / epsilon²)
    where:
        - k is the target dimension,
        - n is the number of samples in the dataset,
        - epsilon is the distortion tolerance.
    """
    if not (0 < epsilon < 1):
        raise ValueError("epsilon must be between 0 and 1")

    if isinstance(X, pd.DataFrame):
        n = X.shape[0]
    elif isinstance(X, np.ndarray):
        n = X.shape[0]
    else:
        raise TypeError("X must be a DataFrame or a NumPy ndarray")

    return int(np.ceil(4 * np.log(n) / (epsilon ** 2)))


def get_jl_output_dim_from_tsfile(filepath: str, epsilon: float) -> int:
    """
    Compute the Johnson-Lindenstrauss backed random projection dimensionality directly from a
    time series dataset stored in '.ts' format.

    This function loads the dataset, determines the number of samples, and
    computes the required projection dimensionality using the JL lemma.

    Parameters
    ----------
    filepath : str
        Path to the `.ts` time series file.
    
    epsilon : float
        Distortion tolerance (0 < epsilon < 1). See `get_jl_output_dim_from_data`.

    Returns
    -------
    int
        The required projection dimensionality according to the JL lemma.
    """
    X, _ = load_from_tsfile_to_dataframe(filepath)
    return get_jl_output_dim_from_data(X, epsilon)