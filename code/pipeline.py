import numpy as np
import matplotlib.pyplot as plt
import time
from sktime.datasets import load_from_tsfile_to_dataframe
from projection import apply_proj
#from sklearn.model_selection import learning_curve
from sklearn.base import clone
from sktime.classification.distance_based import ElasticEnsemble
from sktime.classification.hybrid import HIVECOTEV1
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import cross_val_score
import pandas as pd

def custom_cross_val_score_with_timing(estimator, X, y, cv, scoring='accuracy'):
    """
    Custom version of cross_val_score that measures training and inference times.
    
    Parameters
    ----------
    estimator : object
        The model to use
    X : array-like
        Feature dataset
    y : array-like
        Target labels
    cv : cross-validation generator
        Cross-validation splitting strategy
    
    Returns
    -------
    scores : array
        Validation scores for each fold
    training_times : array  
        Training time for each fold
    inference_times : array  
        Inference time for each fold
    """
    scores = []
    training_times = []
    inference_times = []
    
    for train_idx, test_idx in cv.split(X, y):
        # Split training and test data
        X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        X_test = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Clone the estimator for each fold
        est_clone = clone(estimator)
        
        # Measure training time
        start_time = time.time()
        est_clone.fit(X_train, y_train)
        end_time = time.time()
        
        training_time = end_time - start_time
        training_times.append(training_time)
        
        # Evaluate the model and measure inference time
        start_inference = time.time()
        score = est_clone.score(X_test, y_test)
        inference_time = time.time() - start_inference
        inference_times.append(inference_time)
        scores.append(score)
        
        print(f"Fold training time: {training_time:.3f}s, Score: {score:.3f}")
    
    return np.array(scores), np.array(training_times), np.array(inference_times)

def custom_learning_curve_with_timing(estimator, X, y, train_sizes, cv, scoring='accuracy'):
    """
    Custom version of learning_curve that measures training and inference times.
    
    Parameters
    ----------
    estimator : object
        The model to use
    X : array-like
        Feature dataset
    y : array-like
        Target labels
    train_sizes : array-like
        Relative or absolute numbers of training examples that will be used to generate the learning curve
    cv : cross-validation generator
        Cross-validation splitting strategy
    
    Returns
    -------
    train_sizes_abs : array
        Absolute sizes of the training sets
    train_scores : array
        Training scores for each size and fold
    test_scores : array  
        Test scores for each size and fold
    training_times : array
        Training times for each size and fold
    inference_times : array
        Inference times for each size and fold
    """
    n_samples = len(X)
    train_sizes_abs = np.array([int(size * n_samples) for size in train_sizes])
    
    train_scores = []
    test_scores = []
    training_times = []
    inference_times = []
    
    for train_size in train_sizes_abs:
        train_scores_size = []
        test_scores_size = []
        training_times_size = []
        inference_times_size = []
        
        print(f"\nTraining with {train_size} samples:")

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            np.random.shuffle(train_idx)
            np.random.shuffle(test_idx)
            # Limit training set size
            train_idx_subset = train_idx[:train_size]

            X_train = X.iloc[train_idx_subset] if hasattr(X, 'iloc') else X[train_idx_subset]
            X_test = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
            y_train, y_test = y[train_idx_subset], y[test_idx]
            
            # Clone the estimator
            est_clone = clone(estimator)
            
            # Measure training time
            start_time = time.time()
            est_clone.fit(X_train, y_train)
            end_time = time.time()
            
            training_time = end_time - start_time
            training_times_size.append(training_time)
            
            # Evaluate on training set
            train_score = est_clone.score(X_train, y_train)
            train_scores_size.append(train_score)
            
            # Evaluate on test set
            start_inference = time.time()
            test_score = est_clone.score(X_test, y_test)
            inference_time = time.time() - start_inference
            inference_times_size.append(inference_time)
            test_scores_size.append(test_score)
            
            print(f"  Fold {fold_idx+1}: {training_time:.3f}s, Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        train_scores.append(train_scores_size)
        test_scores.append(test_scores_size)
        training_times.append(training_times_size)
        inference_times.append(inference_times_size)
    
    return (np.array(train_sizes_abs), 
            np.array(train_scores), 
            np.array(test_scores), 
            np.array(training_times),
            np.array(inference_times))

def load_and_combine_datasets(path, path2=None):
    """
    Load and combine time series datasets from .ts files.
    
    Parameters
    ----------
    path : str
        Path to the first .ts file
    path2 : str, optional
        Path to the second .ts file to combine with the first dataset
        
    Returns
    -------
        (X, y): tuple
            Combined (X, y) datasets
        
    Raises
    ------
        ValueError: If datasets have incompatible dimensions
    """
    # Load the first dataset
    X, y = load_from_tsfile_to_dataframe(path)
    
    if path2 is not None:
        X2, y2 = load_from_tsfile_to_dataframe(path2)
        
        # Compatibility checks
        if X.shape[1] != X2.shape[1]:
            raise ValueError(f"Incompatible number of dimensions: {X.shape[1]} vs {X2.shape[1]}")
        
        # Check if column names match
        if not X.columns.equals(X2.columns):
            print("Warning: Column names differ, using position-based concatenation")
        
        # Check for different series lengths (if applicable)
        try:
            if hasattr(X.iloc[0, 0], '__len__') and hasattr(X2.iloc[0, 0], '__len__'):
                len1 = len(X.iloc[0, 0])
                len2 = len(X2.iloc[0, 0])
                if len1 != len2:
                    print(f"Warning: Different series lengths: {len1} vs {len2}")
        except (IndexError, TypeError):
            # Handle cases where series structure is different
            pass
        
        # Check label compatibility
        unique_y1 = set(np.unique(y))
        unique_y2 = set(np.unique(y2))
        if unique_y1 != unique_y2:
            print(f"Info: Different label sets found - Dataset 1: {unique_y1}, Dataset 2: {unique_y2}")
        
        # Concatenation with index reset
        X = pd.concat([X, X2], axis=0, ignore_index=True)
        y = pd.concat([pd.Series(y), pd.Series(y2)], axis=0, ignore_index=True)
        
        print(f"Combined dataset: {len(X)} samples, {X.shape[1]} dimensions")
        print(f"Class distribution: {y.value_counts().to_dict()}")
    
    return X, y

def pipeline(
    path,
    path2=None,
    model="knn",
    dist="euclidean",
    k_neighbors=1,
    lc="y",
    lc_splits=10,
    proj="gaussian",
    eps=0.2,
    cv_splits=5,
    plot="n",
    df_export="n",
    rs=None,
    rs1=12,
    rs2=21,
    rs3=42
):
    """
    Process and evaluate a time series classification pipeline with timing measurements.
    
    This function loads a time series dataset from a '.ts' file, optionally applies a random projection
    (dimensionality reduction), selects and trains a classification model, and evaluates its performance
    using cross-validation and by optionally plotting a learning curve.

    Parameters
    ----------
    path : str
        Path to the '.ts' file containing the time series dataset.
    path2 : str, optional
        Path to a second '.ts' file to combine with the first dataset.
        If provided, the datasets will be concatenated.
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
        Whether to compute a learning curve ("y") or only compute cross-validation scores ("n").
    lc_splits : int, default=10
        Number of training size splits for the learning curve.
    proj : {"gaussian", "sparse", "n"}, default="gaussian"
        Projection method to apply:
            - "gaussian": Gaussian random projection.
            - "sparse": Sparse random projection.
            - "n": No projection is applied.
    eps : float, default=0.2
        Epsilon parameter for the projection step (controls the projection accuracy).
    cv_splits : int, default=5
        Number of folds for Stratified K-Fold cross-validation.
    plot : {"y", "n"}, default="n"
        Whether to plot the learning curve statistics ("y") or only compute and print the values ("n").
    df_export : {"y", "n"}, default="n"
        Whether to return a pandas dataframe containing all the results ("y") or not ("n").
    rs : int or None, default=None
        If provided, overrides rs1, rs2, and rs3 with the same random seed.
    rs1 : int, default=21
        Random seed for the projection step.
    rs2 : int, default=21
        Random seed for the model initialization.
    rs3 : int, default=21
        Random seed for cross-validation splitting.

    Returns
    -------
    df : pandas.DataFrame, optional
        If 'df_export="y"', returns a dataframe containing the results.
        The dataframe includes mean accuracy, mean total time, mean training time,
        mean inference time, and projection time for each training size (if learning curve is computed).
        If 'df_export="n"', returns None.
    
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
    X, y = load_and_combine_datasets(path,path2=path2)

    # Validate projection method
    if proj not in ["gaussian", "sparse", "n"]:
        raise ValueError("projection must be 'gaussian', 'sparse', or 'n' if no projection is wanted, default: 'gaussian'")
    
    # Validate model choice
    if model not in ["knn", "ee", "hivecotev1", "hivecotev2"]:
        raise ValueError("model must be 'knn', 'ee', 'hivecotev1', or 'hivecotev2', default: 'knn'")
    
    # If a single random state is provided, apply it to all components
    if rs is not None:
        rs1 = rs2 = rs3 = rs
    
    # Apply projection if enabled
    if proj != "n":
        start_time=time.time()
        X = apply_proj(X, projection=proj, epsilon=eps, random_state=rs1)
        proj_time=time.time() - start_time
    
    else:
        proj_time=0
    
    # Initialize the chosen model
    if model == "ee":
        est = ElasticEnsemble(random_state=rs2)
    if model == "hivecotev1":
        est = HIVECOTEV1(random_state=rs2)
    elif model == "hivecotev2":
        est = HIVECOTEV2(random_state=rs2)
    else:  # "knn"
        est = KNeighborsTimeSeriesClassifier(
            n_neighbors=k_neighbors,
            distance=dist
        )
    
    # Define cross-validation strategy
    crv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=rs3)
    
    # If learning curve is requested
    if lc == "y":
        # Compute learning curve data with timing
        train_sizes, train_scores, test_scores, training_times, inference_times = custom_learning_curve_with_timing(
            estimator=est,
            X=X,
            y=y,
            train_sizes=np.linspace(0.1, 1.0, lc_splits),
            cv=crv,
            scoring="accuracy"
        )

        mean_training_times = np.mean(training_times, axis=1)
        mean_accuracy_scores = np.mean(test_scores, axis=1)
        mean_inference_times = np.mean(inference_times, axis=1)
        std_training_times = np.std(training_times, axis=1)

        if plot!="n":
            # Plot training and validation accuracy curves
            plt.figure(figsize=(12, 4))
        
            plt.subplot(1, 2, 1)
            plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Training accuracy")
            plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Validation accuracy")
            plt.xlabel("Training set size")
            plt.ylabel("Accuracy")
            plt.title("Learning curve")
            plt.legend()
            plt.grid()
        
            # Plot training times
            plt.subplot(1, 2, 2)
            plt.plot(train_sizes, mean_training_times, 'r-', label="Mean training time")
            plt.fill_between(train_sizes, mean_training_times - std_training_times, mean_training_times + std_training_times, alpha=0.3)
            plt.xlabel("Training set size")
            plt.ylabel("Training time (seconds)")
            plt.title("Training time curve")
            plt.legend()
            plt.grid()
        
            plt.tight_layout()
            plt.show()

        # Display detailed statistics
        print(f"\n=== LEARNING CURVE RESULTS ===")
        print(f"Test accuracy scores: {test_scores}")
        print(f"Mean test accuracy: {test_scores.mean():.3f}")
        print(f"Training times (seconds): {training_times}")
        print(f"Mean training time per fold: {training_times.mean():.3f}s")
        print(f"Total training time: {training_times.sum():.3f}s")
        print(f"Min/Max training time: {training_times.min():.3f}s / {training_times.max():.3f}s")

        # Export a pandas dataframe containing the results
        if df_export=="y":
            columns = [f"{col} {i+1}"
                    for i in range(len(mean_training_times))
                    for col in ["mean_accuracy", "mean_total_time", "mean_training_time", "mean_inference_time", "projection_time"]]
            
            proj_time_array = np.full_like(mean_training_times, proj_time)

            df = pd.DataFrame(index=range(1), columns=columns)
            df.loc[0]=[elt[i]
                for i in range(len(mean_training_times))
                for elt in [mean_accuracy_scores,np.array(mean_training_times)+np.array(mean_inference_times)+proj_time,mean_training_times,mean_inference_times,proj_time_array]
            ]
            
            return(df)
    
    else:
        # Perform standard cross-validation with timing
        print("=== CROSS-VALIDATION WITH TIMING ===")
        scores, training_times, inference_times = custom_cross_val_score_with_timing(
            estimator=est,
            X=X,
            y=y,
            cv=crv,
            scoring='accuracy'
        )

        # Display accuracy and timing statistics
        print(f"\n=== CROSS-VALIDATION RESULTS ===")
        print(f"Accuracy scores: {scores}")
        print(f"Mean accuracy: {scores.mean():.3f}")
        print(f"Training times (seconds): {training_times}")
        print(f"Mean training time: {training_times.mean():.3f}s")
        print(f"Total training time: {training_times.sum():.3f}s")
        print(f"Min/Max training time: {training_times.min():.3f}s / {training_times.max():.3f}s")

        # Export a pandas dataframe containing the results
        if df_export=="y":
            df=pd.DataFrame(index=range(0),columns=[f"mean_accuracy","mean_total_time","mean_training_time","mean_inference_time","projection_time"])
            df.loc[0]=[scores.mean(),training_times.mean()+inference_times.mean()+proj_time,training_times.mean(),inference_times.mean(),proj_time]
            return(df)