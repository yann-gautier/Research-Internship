from pipeline import pipeline
import pandas as pd

paths = []

df = pd.concat([
    pipeline(paths[0],model="knn",dist="dtw",k_neighbors=1,proj="gaussian",cv_splits=5,df_export="n"),
    pipeline(paths[0],model="knn",dist="dtw",k_neighbors=3,proj="gaussian",cv_splits=5,df_export="n"),
    pipeline(paths[0],model="knn",dist="dtw",k_neighbors=5,proj="gaussian",cv_splits=5,df_export="n"),
    pipeline(paths[0],model="knn",dist="dtw",k_neighbors=7,proj="gaussian",cv_splits=5,df_export="n"),
    pipeline(paths[0],model="knn",dist="dtw",k_neighbors=9,proj="gaussian",cv_splits=5,df_export="n"),
    pipeline(paths[0],model="knn",dist="dtw",k_neighbors=11,proj="gaussian",cv_splits=5,df_export="n"),
    pipeline(paths[0],model="knn",dist="dtw",k_neighbors=15,proj="gaussian",cv_splits=5,df_export="n"),
    pipeline(paths[0],model="knn",dist="dtw",k_neighbors=21,proj="gaussian",cv_splits=5,df_export="n"),
    pipeline(paths[0],model="knn",dist="dtw",k_neighbors=31,proj="gaussian",cv_splits=5,df_export="n")
]
)