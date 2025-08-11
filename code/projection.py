from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from jl_adjustment import get_jl_output_dim_from_data
from sktime.datatypes._panel._convert import from_nested_to_2d_array
import pandas as pd

def apply_proj(X, projection="gaussian", epsilon=0.2, random_state=21):
    """
    Apply a Johnson-Lindenstrauss backed random projection to multivariate time series data.

    This function:
      1. Converts multivariate time series data into a 2D tabular format.
      2. Determines the optimal number of projection dimensions using the JL lemma.
      3. Applies either a Gaussian or a Sparse random projection.
      4. Returns the projected data in a univariate DataFrame format.

    Parameters
    ----------
    X : pd.DataFrame or sktime-compatible panel data
        The input multivariate time series data. It must be compatible with
        'sktime.transformations.panel.Tabularizer'.
    
    projection : {"gaussian", "sparse"}, default="gaussian"
        Type of random projection to use:
        - "gaussian" → Gaussian Random Projection.
        - "sparse"   → Sparse Random Projection.
    
    epsilon : float, default=0.2
        Tolerance parameter for the JL lemma. Controls the quality of the projection:
        smaller values preserve pairwise distances more accurately, but require more dimensions.
    
    random_state : int, default=21
        Seed for random number generation to ensure reproducibility.

    Returns
    -------
    X_univariate : pd.DataFrame
        A DataFrame with a single column "dim_0", where each row contains
        a pandas Series representing the projected feature vector for a single instance.
    
    Notes
    -----
    - This function uses 'get_jl_output_dim_from_data' from the jl_adjustment module to compute the 
      projection dimension required for the specified 'epsilon'.
    - The transformation ensures that distances between data points are approximately
      preserved according to the JL lemma.
    """

    # Convert multivariate time series data to tabular format
    X_tab = from_nested_to_2d_array(X)

    # Determine number of components using the JL lemma
    n_components = get_jl_output_dim_from_data(X_tab, epsilon)

    # Choose and initialize the projector
    if projection == "gaussian":
        projector = GaussianRandomProjection(n_components=n_components, random_state=random_state)
    else:
        projector = SparseRandomProjection(n_components=n_components, random_state=random_state)
    
    # Apply the projection
    X_proj = projector.fit_transform(X_tab)

    # Wrap projected vectors into a univariate DataFrame format
    X_univariate = pd.DataFrame({
        'dim_0': [pd.Series(row) for row in X_proj]
    })

    return X_univariate