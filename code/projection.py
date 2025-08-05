from sklearn.random_projection import GaussianRandomProjection

def random_projection(X_train, X_test, output_dim, seed=None):
    projector = GaussianRandomProjection(n_components=output_dim, random_state=seed)
    X_train_proj = projector.fit_transform(X_train)
    X_test_proj = projector.fit_transform(X_test)
    return X_train_proj, X_test_proj 
