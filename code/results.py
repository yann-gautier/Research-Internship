from pipeline import pipeline
import pandas as pd

def concat(path,path2=None):
    """Generate all experiment combinations for a single dataset"""
    results_list = []
    
    # Configuration
    k_values = [1]
    projections = ["gaussian"]
    
    # KNN with different neighbors values and projections
    for proj in projections:
        for k in k_values:
            df = pipeline(
                path=path,
                path2=path2, 
                model="knn", 
                dist="euclidean", 
                k_neighbors=k, 
                proj=proj, 
                cv_splits=5, 
                df_export="y"
            )
            results_list.append(df)
        
        # Elastic Ensemble for each projection
        # df = pipeline(
        #     path=path,
        #     path2=path2, 
        #     model="ee",
        #     proj=proj,
        #     cv_splits=5,
        #     df_export="y"
        # )
        # results_list.append(df)
    
    return pd.concat(results_list, axis=1)

def results(paths):
    """Process multiple datasets and export results"""
    print(f"Processing {len(paths)} datasets...")
    
    all_results = []
    for i in range(0, len(paths), 2):
        print(f"[{i/2+1}/{len(paths)/2}] Processing: {paths[i]} and {paths[i+1]}")
        try:
            df = concat(paths[i],paths[i+1])
            all_results.append(df)
        except Exception as e:
            print(f"Error processing {paths[i]} and {paths[i+1]}: {e}")
            print("Continuing with next dataset pair...")
            continue
    
    if not all_results:
        print("No datasets were processed successfully!")
        return None
    
    final_df = pd.concat(all_results, axis=0, ignore_index=True)
    
    # Export
    final_df.to_csv('results.csv', index=False, encoding='utf-8')
    print(f"Results exported to 'results.csv' ({final_df.shape[0]} rows, {final_df.shape[1]} columns)")
    
    return final_df