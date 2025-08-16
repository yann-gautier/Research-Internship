import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def merge_timeseries(train_file, test_file, output_file, add_split_column=False):
    """
    Merge train and test time series files into a single dataset.
    
    Args:
        train_file (str): Path to the training file
        test_file (str): Path to the test file  
        output_file (str): Output file path
        add_split_column (bool): Whether to add a column indicating data origin (train/test)
    
    Returns:
        pandas.DataFrame: Merged dataset or None if error occurred
    """
    
    try:
        # Read files with automatic format detection
        def read_ts_file(filepath):
            """
            Read a time series file by attempting different formats.
            
            Args:
                filepath (str): Path to the file to read
                
            Returns:
                pandas.DataFrame: Loaded data or None if failed
            """
            try:
                # Try CSV with different separators
                for sep in [',', '\t', ' ', ';']:
                    try:
                        df = pd.read_csv(filepath, sep=sep)
                        if len(df.columns) > 1:  # If multiple columns, likely correct format
                            print(f"File {filepath} read with separator '{sep}'")
                            return df
                    except:
                        continue
                
                # If previous attempts failed, try line-by-line reading
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                
                # Detect format from first line
                first_line = lines[0].strip()
                if ',' in first_line:
                    sep = ','
                elif '\t' in first_line:
                    sep = '\t'
                else:
                    sep = ' '
                
                return pd.read_csv(filepath, sep=sep)
                
            except Exception as e:
                print(f"Error reading file {filepath}: {e}")
                return None
        
        # Read input files
        print("Reading training file...")
        train_data = read_ts_file(train_file)
        
        print("Reading test file...")
        test_data = read_ts_file(test_file)
        
        if train_data is None or test_data is None:
            raise ValueError("Unable to read one or more input files")
        
        print(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
        
        # Validate column compatibility
        if not train_data.columns.equals(test_data.columns):
            print("Warning: Column names do not match exactly between train and test files")
            print(f"Train columns: {list(train_data.columns)}")
            print(f"Test columns: {list(test_data.columns)}")
        
        # Add split identifier column if requested
        if add_split_column:
            train_data['split'] = 'train'
            test_data['split'] = 'test'
        
        # Merge datasets
        merged_data = pd.concat([train_data, test_data], ignore_index=True)
        
        # Add temporal index if no explicit time column exists
        time_columns = [col for col in merged_data.columns 
                       if any(keyword in col.lower() for keyword in ['time', 'timestamp', 'date'])]
        
        if not time_columns:
            merged_data['time_index'] = range(len(merged_data))
            print("Added temporal index column")
        
        # Save merged dataset
        output_path = Path(output_file)
        # Use tab separator for .ts files, comma for others
        sep = '\t' if output_path.suffix == '.ts' else ','
        merged_data.to_csv(output_file, sep=sep, index=False)
        
        print(f"Merge completed successfully: {output_file}")
        print(f"Final dataset shape: {merged_data.shape}")
        print(f"Columns: {list(merged_data.columns)}")
        
        # Display data summary
        print("Dataset preview:")
        print(merged_data.head())
        print(f"\nDataset info:")
        print(merged_data.info())
        
        return merged_data
        
    except Exception as e:
        print(f"Merge operation failed: {e}")
        return None



def validate_timeseries_data(data):
    """
    Perform basic validation on time series data.
    
    Args:
        data (pandas.DataFrame): Time series dataset to validate
        
    Returns:
        dict: Validation results and statistics
    """
    validation_results = {
        'shape': data.shape,
        'missing_values': data.isnull().sum().sum(),
        'duplicate_rows': data.duplicated().sum(),
        'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(data.select_dtypes(include=['object']).columns),
    }
    
    # Check for temporal ordering if time column exists
    time_columns = [col for col in data.columns 
                   if any(keyword in col.lower() for keyword in ['time', 'timestamp', 'date'])]
    
    if time_columns:
        time_col = time_columns[0]
        try:
            data[time_col] = pd.to_datetime(data[time_col])
            validation_results['temporal_column'] = time_col
            validation_results['time_range'] = (data[time_col].min(), data[time_col].max())
            validation_results['is_sorted'] = data[time_col].is_monotonic_increasing
        except:
            validation_results['temporal_parsing_error'] = True
    
    return validation_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Merge time series datasets for cross-validation pipelines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_timeseries.py --train train.ts --test test.ts --output merged.ts
  python merge_timeseries.py --train train.csv --test test.csv --output merged.csv --no-split-column
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', help='Training dataset file path')
    
    parser.add_argument('--test', required=True, help='Test dataset file path')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--no-split-column', action='store_true', 
                       help='Do not add split identifier column')
    parser.add_argument('--validate', action='store_true',
                       help='Perform data validation after merging')
    
    args = parser.parse_args()
    
    # Execute merge operation
    merged_data = merge_timeseries(
        train_file=args.train,
        test_file=args.test,
        output_file=args.output,
        add_split_column=not args.no_split_column
    )
    
    # Perform validation if requested
    if args.validate and merged_data is not None:
        print("Performing data validation...")
        validation_results = validate_timeseries_data(merged_data)
        
        print("\n" + "="*50)
        print("DATA VALIDATION RESULTS")
        print("="*50)
        for key, value in validation_results.items():
            print(f"{key}: {value}")
        print("="*50)