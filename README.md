# Research-Internship

This project explores supervised classification of time series with a focus on dimensionality reduction via random projections.
It provides a configurable and modular framework for experimenting with different parameters, enabling reproducible benchmarks.
The study also includes an in-depth analysis of learning curves to understand the trade-offs between model complexity, accuracy, and data size.

In our experiments, we evaluate k-values = [1] for the nearest neighbors using DTW as the distance measure and the EE model. We consider three settings: Gaussian projection, sparse projection, and no projection. This results in 6 distinct model configurations, each assessed across 10 different training set sizes, leading to 60 trained models in total. Each configuration is further validated with 5-fold cross-validation, amounting to 300 training runs overall.

## Code

- [pipeline.py](code/pipeline.py) : the pipeline with all parameters
- [projection.py](code/projection.py) : the function that applies the random projection to the times series dataset
- [jl_adjustment.py](code/jl_adjustment.py) : the function that computes the size of the reduced dimensionality

### Requirements

- Python
- numpy
- pandas
- sktime
- scikit-learn
- matplotlib

### Using the pipeline

[main.py](code/main.py)

```bash
Arguments:
-p --path        : Path to the .ts file containing the dataset
--path2          : Path to the second .ts file containing the dataset to concatenate with the first dataset (optional, default None)
-m --model       : Type of model to use (optional, default knn)
-d --dist        : Distance metric for KNN (optional, default euclidean)
-n --k_neighbors : Number of neighbors for KNN (optional, default 1)
-l --lc          : Compute learning curve (y) or not (n) (optional, default 'y')
--lc_splits      : Number of points for the learning curve (optional, default 10)
--proj           : Type of random projection (optional, default 'gaussian')
--eps            : Epsilon parameter for the distortion accepted through projection (optional, default 0.2)
--cv_splits      : Number of folds for cross-validation (optional, default 5)
--plot           : Plot the learning curve (y) or not (n) (optional, default 'n')
--df_export      : Return a DataFrame with the results (optional, default='n')
-r --rs          : Global random state (applied to all components) (optional, default None)
--rs1            : Random state for projection (optional, default 12)
--rs2            : Random state for the classifier (optional, default 21)
--rs3            : Random state for StratifiedKFold (optional, default 42)

Examples:
> python main.py --path ./input/worms/Worms_TRAIN.ts --path2 ./input/worms/Worms_TEST.ts --model knn --dist dtw --plot y
> python main.py --path ./input/worms/Worms_TRAIN.ts --path2 ./input/worms/Worms_TEST.ts --model ee --lc n --proj sparse --rs1 42
```

### Reproducing the Experiments

[reproducibility.py](code/reproducibility.py)

```bash
Arguments:
-p --paths : dataset paths

Example:
> python ./input/reproducibility.py --paths ./input/worms/Worms_TRAIN.ts ./input/worms/Worms_TEST.ts
```

The script implementing the experimentations is stored in [results.py](code/results.py).
It returns a csv file with one line for a dataset and each model used adds 6 columns : mean_accuracy, mean_total_time, mean_training_time, mean_inference_time, projection_time, reduced_dimension

## To be done
