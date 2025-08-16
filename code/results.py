from pipeline import pipeline
import pandas as pd

def concat(path):
    """Generate all experiment combinations for a single dataset"""
    results_list = []
    
    # Configuration
    k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31]
    projections = ["gaussian", "sparse", "no"]
    
    # KNN with different neighbors values and projections
    for proj in projections:
        for k in k_values:
            df = pipeline(
                path=path, 
                model="knn", 
                dist="dtw", 
                k_neighbors=k, 
                proj=proj, 
                cv_splits=5, 
                df_export="y"
            )
            results_list.append(df)
        
        # Elastic Ensemble for each projection
        df = pipeline(
            path=path, 
            model="ee", 
            proj=proj, 
            cv_splits=5, 
            df_export="y"
        )
        results_list.append(df)
    
    return pd.concat(results_list, axis=1)

def results(paths):
    """Process multiple datasets and export results"""
    print(f"Processing {len(paths)} datasets...")
    
    all_results = []
    for i, path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] Processing: {path}")
        df = concat(path)
        all_results.append(df)
    
    final_df = pd.concat(all_results, axis=0, ignore_index=True)
    
    # Export
    final_df.to_csv('results.csv', index=False, encoding='utf-8')
    print(f"Results exported to 'results.csv' ({final_df.shape[0]} rows, {final_df.shape[1]} columns)")
    
    return final_df