from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from jl_adjustment import get_jl_output_dim_from_data
from sktime.datatypes._panel._convert import from_nested_to_2d_array
import pandas as pd

def apply_proj(X, X_test=None, projection="gaussian", epsilon=0.2, random_state=21):
    """
    Apply a Johnson-Lindenstrauss backed random projection to multivariate time series data.

    This function:
        Converts multivariate time series data into a 2D tabular format.
        Determines the optimal number of projection dimensions using the JL lemma.
        Applies either a Gaussian or a Sparse random projection.
        Returns the projected data in a univariate DataFrame format.

    Parameters
    ----------
    X : pd.DataFrame or sktime-compatible panel data
        The input multivariate time series data. Each column represents a variable,
        and each cell contains a pandas Series representing the time series for that variable.
    
    X_test : pd.DataFrame or sktime-compatible panel data, optional (default=None)
        Optional test dataset to be projected using the same transformation as 'X'.
    
    projection : {"gaussian", "sparse"}, default="gaussian"
        Type of random projection to use:
        - "gaussian" : Gaussian Random Projection.
        - "sparse"   : Sparse Random Projection.
    
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
    
    X_test_univariate : pd.DataFrame, optional
        If 'X_test' is provided, this DataFrame contains the projected feature vectors
    
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
    
    if X_test is not None:
        X_test_tab = from_nested_to_2d_array(X_test)
        # Fit on training data and transform both training and test data
        X_proj = projector.fit_transform(X_tab)
        X_test_proj = projector.transform(X_test_tab)

        # Wrap projected vectors into a univariate DataFrame format
        X_univariate = pd.DataFrame({
            'dim_0': [pd.Series(row) for row in X_proj]
        })
        X_test_univariate = pd.DataFrame({
            'dim_0': [pd.Series(row) for row in X_test_proj]
        })

        return X_univariate, X_test_univariate, n_components
    
    # Apply the projection
    X_proj = projector.fit_transform(X_tab)

    # Wrap projected vectors into a univariate DataFrame format
    X_univariate = pd.DataFrame({
        'dim_0': [pd.Series(row) for row in X_proj]
    })

    return X_univariate, n_components