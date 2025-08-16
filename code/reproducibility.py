from pipeline import pipeline
import pandas as pd

paths = []

def concat(path):
    df = pd.concat([
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=1,proj="gaussian",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=3,proj="gaussian",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=5,proj="gaussian",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=7,proj="gaussian",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=9,proj="gaussian",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=11,proj="gaussian",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=15,proj="gaussian",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=21,proj="gaussian",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=31,proj="gaussian",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="ee",proj="gaussian",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=1,proj="sparse",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=3,proj="sparse",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=5,proj="sparse",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=7,proj="sparse",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=9,proj="sparse",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=11,proj="sparse",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=15,proj="sparse",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=21,proj="sparse",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=31,proj="sparse",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="ee",proj="sparse",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=1,proj="no",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=3,proj="no",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=5,proj="no",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=7,proj="no",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=9,proj="no",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=11,proj="no",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=15,proj="no",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=21,proj="no",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="knn",dist="dtw",k_neighbors=31,proj="no",cv_splits=5,df_export="y"),
        pipeline(paths[0],model="ee",proj="no",cv_splits=5,df_export="y")
    ], axis = 1
    )
    return df

df = pd.concat([concat(path) for path in paths])

df.to_csv('results.csv',
          index=False,
          sep=',',
          encoding='utf-8',
          header=True) 