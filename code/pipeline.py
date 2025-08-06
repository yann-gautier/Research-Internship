from sklearn.pipeline import Pipeline
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neighbors import KNeighborsClassifier
from jl_adjustment import get_jl_output_dim_from_data

def get_pipeline_from_data(X, epsilon=0.3, n_neighbors=1, random_state=42):
    n_components = get_jl_output_dim_from_data(X, epsilon)
    projector = GaussianRandomProjection(n_components=n_components, random_state=random_state)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    return Pipeline([
        ("projection", projector),
        ("1nn", knn)
    ])
