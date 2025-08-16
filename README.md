# Research-Internship

This project explores supervised classification of time series with a focus on dimensionality reduction via random projections.
It provides a configurable and modular framework for experimenting with different parameters, enabling reproducible benchmarks.
The study also includes an in-depth analysis of learning curves to understand the trade-offs between model complexity, accuracy, and data size.

To be documented: the returns of the process function, with thee pandas dataframe structure

To be done: the reproducibility file enabling to generate our results

In our experiments, we evaluate k-values = [1, 3, 5, 7, 9, 11, 15, 21, 31] for the nearest neighbors using DTW as the distance measure and the EE model. We consider three settings: Gaussian projection, sparse projection, and no projection. This results in 28 distinct model configurations, each assessed across 10 different training set sizes, leading to 300 trained models in total. Each configuration is further validated with 5-fold cross-validation, amounting to 1,500 training runs overall.
