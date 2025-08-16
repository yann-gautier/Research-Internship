# Research-Internship

This project explores supervised classification of time series with a focus on dimensionality reduction via random projections.
It provides a configurable and modular framework for experimenting with different parameters, enabling reproducible benchmarks.
The study also includes an in-depth analysis of learning curves to understand the trade-offs between model complexity, accuracy, and data size.

To be documented: the returns of the process function, with thee pandas dataframe structure

To be done: the reproducibility file enabling to generate our results

Dans nos expérimentations, on va tester k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31] pour les plus proches voisins, distance : dtw, et modèle ee. On teste avec projection gaussienne, pojection creuse, sans projection. Ce qui fait 28 modèles, avec 10 tailles différentes de données d'entraînement, donc 280 modèles entraînés, fois 5 cross validations. Donc 1400 entraînements.