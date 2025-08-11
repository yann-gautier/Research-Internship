import numpy as np
import matplotlib.pyplot as plt
from sktime.datasets import load_from_tsfile_to_dataframe
from projection import apply_proj
from sklearn.model_selection import learning_curve
#from sktime.classification.dictionary_based import ElasticEnsemble
from sktime.classification.hybrid import HIVECOTEV1
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

def process(
    path,
    model="knn",
    distance="euclidean",
    n_neighbors=1,
    lc="y",
    lc_splits=10,
    proj="gaussian",
    eps=0.2,
    cv_splits=5,
    rs=None,
    rs1=21,
    rs2=21,
    rs3=21
):
    """
    Process and evaluate a time series classification pipeline.

    This function loads a time series dataset from a '.ts' file, optionally applies a random projection
    (dimensionality reduction), selects and trains a classification model, and evaluates its performance
    using cross-validation and by optionally plotting a learning curve.

    Parameters
    ----------
    path : str
        Path to the '.ts' file containing the time series dataset.
    model : {"knn", "ee", "hivecotev1", "hivecotev2"}, default="knn"
        Classification model to use:
            - "knn": k-Nearest Neighbors classifier for time series.
            - "ee": Elastic Ensemble classifier.
            - "hivecotev1": HIVE-COTE version 1 classifier.
            - "hivecotev2": HIVE-COTE version 2 classifier.
    distance : str, default="euclidean"
        Distance metric for k-NN (only applicable when 'model="knn"').
    n_neighbors : int, default=1
        Number of neighbors for k-NN (only applicable when 'model="knn"').
    lc : {"y", "n"}, default="y"
        Whether to plot a learning curve ("y") or only display cross-validation scores ("n").
    lc_splits : int, default=10
        Number of training size splits for the learning curve.
    proj : {"gaussian", "sparse", "no"}, default="gaussian"
        Projection method to apply:
            - "gaussian": Gaussian random projection.
            - "sparse": Sparse random projection.
            - "no": No projection is applied.
    eps : float, default=0.2
        Epsilon parameter for the projection step (controls the projection accuracy).
    cv_splits : int, default=5
        Number of folds for Stratified K-Fold cross-validation.
    rs : int or None, default=None
        If provided, overrides rs1, rs2, and rs3 with the same random seed.
    rs1 : int, default=21
        Random seed for the projection step.
    rs2 : int, default=21
        Random seed for the model initialization.
    rs3 : int, default=21
        Random seed for cross-validation splitting.

    Raises
    ------
    ValueError
        If 'proj' or 'model' is not one of the allowed values.

    Notes
    -----
    - This function supports classification of univariate or multivariate time series datasets.
    - The 'apply_proj' function is implemented in the 'projection.py' module.
    """

    # Load dataset from the given .ts file path
    X, y = load_from_tsfile_to_dataframe(path)

    # Validate projection method
    if proj not in ["gaussian", "sparse", "no"]:
        raise ValueError("projection must be 'gaussian', 'sparse', or 'no' if no projection is wanted, default: 'gaussian'")
    
    # Validate model choice
    if model not in ["knn", "ee", "hivecotev1", "hivecotev2"]:
        raise ValueError("model must be 'knn', 'ee', 'hivecotev1', or 'hivecotev2', default: 'knn'")
    
    # If a single random state is provided, apply it to all components
    if rs is not None:
        rs1 = rs2 = rs3 = rs
    
    # Apply projection if enabled
    if proj != "no":
        X = apply_proj(X, projection=proj, epsilon=eps, random_state=rs1)
    
    # Initialize the chosen model
    # if model == "ee":
    #     est = ElasticEnsemble(random_state=rs2)
    if model == "hivecotev1":
        est = HIVECOTEV1(random_state=rs2)
    elif model == "hivecotev2":
        est = HIVECOTEV2(random_state=rs2)
    else:  # "knn"
        est = KNeighborsTimeSeriesClassifier(
            n_neighbors=n_neighbors,
            distance=distance
        )
    
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=rs3)
    
    # If learning curve is requested
    if lc == "y":
        # Compute learning curve data
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=est,
            X=X,
            y=y,
            train_sizes=np.linspace(0.1, 1.0, lc_splits),
            cv=cv,
            scoring="accuracy"
        )

        # Plot training and validation accuracy curves
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Training accuracy")
        plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Validation accuracy")
        plt.xlabel("Training set size")
        plt.ylabel("Accuracy")
        plt.title("Learning curve:")
        plt.legend()
        plt.grid()
        plt.show()

        # Display test scores for each split
        print(f"Accuracy scores: {test_scores}")
    
    else:
        # Perform standard cross-validation without plotting
        scores = cross_val_score(
            estimator=est,
            X=X,
            y=y,
            cv=cv,
            scoring='accuracy'
        )

        # Display accuracy statistics
        print(f"Accuracy scores: {scores}")
        print(f"Mean accuracy: {scores.mean():.3f}")