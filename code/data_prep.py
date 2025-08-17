import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import zipfile
import tarfile
import shutil
import tempfile
import os
from sktime.datasets import load_from_tsfile_to_dataframe

def write_ts_file(X, y, filepath):
    """
    Write time series data to .ts file format.
    
    Args:
        X (pd.DataFrame): Time series data (with lists/arrays in cells)
        y (pd.Series or None): Labels
        filepath (str): Output file path
    """
    with open(filepath, 'w') as f:
        # Write problem name
        problem_name = Path(filepath).stem
        f.write(f"@problemName {problem_name}\n")
        
        # Write timestamp info
        f.write("@timeStamps false\n")
        f.write("@missing false\n")
        f.write("@univariate false\n")
        
        # Write dimensions
        f.write(f"@dimensions {len(X.columns)}\n")
        
        # Write equal length (assume all series have same length)
        if len(X) > 0:
            first_series = X.iloc[0, 0]
            if isinstance(first_series, (list, np.ndarray)):
                series_length = len(first_series)
            else:
                series_length = 1
            f.write(f"@equalLength true\n")
            f.write(f"@seriesLength {series_length}\n")
        
        # Write class labels if they exist
        if y is not None and not y.isna().all():
            # Filter out None/NaN values for unique classes
            valid_labels = y.dropna()
            if len(valid_labels) > 0:
                unique_classes = sorted(valid_labels.unique())
                class_labels = ','.join(map(str, unique_classes))
                f.write(f"@classLabel true {class_labels}\n")
            else:
                f.write("@classLabel false\n")
        else:
            f.write("@classLabel false\n")
        
        f.write("@data\n")
        
        # Write data
        for idx in range(len(X)):
            row_data = []
            for col in X.columns:
                series = X.iloc[idx, X.columns.get_loc(col)]
                if isinstance(series, (list, np.ndarray)):
                    # Convert to comma-separated string
                    series_str = ','.join(map(str, series))
                else:
                    series_str = str(series)
                row_data.append(series_str)
            
            # Add label if exists and is not None/NaN
            if y is not None and idx < len(y) and pd.notna(y.iloc[idx]):
                row_data.append(str(y.iloc[idx]))
            
            f.write(':'.join(row_data) + '\n')

def convert_arrays_to_lists(df):
    """
    Convert numpy arrays to lists in DataFrame cells.
    
    Args:
        df (pd.DataFrame): DataFrame with potential numpy arrays in cells
        
    Returns:
        pd.DataFrame: DataFrame with arrays converted to lists
    """
    # Create a completely new DataFrame to avoid pandas indexing issues
    new_data = {}
    
    for col in df.columns:
        new_column = []
        for idx in range(len(df)):
            cell_value = df.iloc[idx, df.columns.get_loc(col)]
            if isinstance(cell_value, np.ndarray):
                new_column.append(cell_value.tolist())
            else:
                new_column.append(cell_value)
        new_data[col] = new_column
    
    # Create new DataFrame with converted data
    result_df = pd.DataFrame(new_data, index=df.index)
    return result_df

def merge_timeseries(train_file, test_file, output_file):
    """
    Merge train and test time series files into a single .ts dataset using sktime.
    
    Args:
        train_file (str): Path to the training .ts file
        test_file (str): Path to the test .ts file  
        output_file (str): Output .ts file path
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    try:
        # Read files using sktime
        train_data, train_labels = load_from_tsfile_to_dataframe(train_file)
        test_data, test_labels = load_from_tsfile_to_dataframe(test_file)
        
        print(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
        
        # Convert numpy arrays to lists for proper concatenation
        train_data_converted = convert_arrays_to_lists(train_data)
        test_data_converted = convert_arrays_to_lists(test_data)
        
        # Merge datasets
        merged_data = pd.concat([train_data_converted, test_data_converted], ignore_index=True)
        
        # Helper function to convert labels to pandas Series
        def ensure_series(labels, name='class_vals'):
            if labels is None:
                return None
            elif isinstance(labels, np.ndarray):
                return pd.Series(labels, name=name)
            elif isinstance(labels, pd.Series):
                return labels
            else:
                # Convert other types to Series
                return pd.Series(labels, name=name)
        
        # Merge labels if they exist
        merged_labels = None
        if train_labels is not None and test_labels is not None:
            # Convert both to Series if needed
            train_series = ensure_series(train_labels, 'class_vals')
            test_series = ensure_series(test_labels, 'class_vals')
            merged_labels = pd.concat([train_series, test_series], ignore_index=True)
        elif train_labels is not None:
            # If only train has labels, create None labels for test
            train_series = ensure_series(train_labels, 'class_vals')
            test_none_series = pd.Series([None] * len(test_data), name='class_vals')
            merged_labels = pd.concat([train_series, test_none_series], ignore_index=True)
        elif test_labels is not None:
            # If only test has labels, create None labels for train
            train_none_series = pd.Series([None] * len(train_data), name='class_vals')
            test_series = ensure_series(test_labels, 'class_vals')
            merged_labels = pd.concat([train_none_series, test_series], ignore_index=True)
        
        # Save merged dataset in .ts format
        write_ts_file(merged_data, merged_labels, output_file)
        
        print(f"Merge completed successfully: {output_file}")
        print(f"Final dataset shape: {merged_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"Merge operation failed for {Path(train_file).parent.name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def extract_archive(archive_path, extract_to):
    """
    Extract compressed archive to specified directory.
    
    Args:
        archive_path (str): Path to the compressed file
        extract_to (str): Directory to extract to
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        archive_path = Path(archive_path)
        
        if archive_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix.lower() in ['.tar', '.tar.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"Unsupported archive format: {archive_path.suffix}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Failed to extract {archive_path}: {e}")
        return False

def find_ts_pairs(directory):
    """
    Find train/test pairs of .ts files in a directory.
    Files ending with TRAIN.ts are paired with files ending with TEST.ts
    with the same base name.
    
    Args:
        directory (str): Directory to search
        
    Returns:
        tuple: (train_file, test_file, base_name) or None if no valid pair found
    """
    ts_files = list(Path(directory).glob("*.ts"))
    
    if len(ts_files) < 2:
        return None
    
    # Find files ending with TRAIN.ts and TEST.ts
    train_files = [f for f in ts_files if f.name.endswith('TRAIN.ts')]
    test_files = [f for f in ts_files if f.name.endswith('TEST.ts')]
    
    if not train_files or not test_files:
        print(f"Warning: No files ending with TRAIN.ts or TEST.ts found in {directory}")
        return None
    
    # Match pairs by base name
    for train_file in train_files:
        # Get base name by removing 'TRAIN.ts'
        base_name = train_file.name[:-8]  # Remove 'TRAIN.ts'
        
        # Look for corresponding test file
        test_file_name = f"{base_name}TEST.ts"
        test_file_path = Path(directory) / test_file_name
        
        if test_file_path in test_files:
            return (str(train_file), str(test_file_path), base_name)
    
    print(f"Warning: No matching TRAIN/TEST pairs found in {directory}")
    return None

def batch_merge_compressed_datasets(data_directory, output_directory=None, keep_extracted=False):
    """
    Process all compressed files in a directory and merge train/test pairs.
    
    Args:
        data_directory (str): Directory containing compressed files
        output_directory (str): Directory to save merged files (optional, defaults to data_directory/merged)
        keep_extracted (bool): Whether to keep extracted files after processing
    """
    
    data_dir = Path(data_directory)
    
    # Set default output directory if not provided
    if output_directory is None:
        output_dir = data_dir / "merged"
    else:
        output_dir = Path(output_directory)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all compressed files
    compressed_files = []
    for pattern in ['*.zip', '*.tar', '*.tar.gz', '*.tgz']:
        compressed_files.extend(data_dir.glob(pattern))
    
    if not compressed_files:
        print("No compressed files found in the specified directory")
        return
    
    print(f"Found {len(compressed_files)} compressed files to process")
    
    successful_merges = 0
    failed_merges = 0
    
    # Create temporary directory for extractions
    with tempfile.TemporaryDirectory() as temp_dir:
        
        for archive_file in compressed_files:
            print(f"\nProcessing: {archive_file.name}")
            
            # Create unique extraction directory
            extract_dir = Path(temp_dir) / archive_file.stem
            extract_dir.mkdir(exist_ok=True)
            
            # Extract archive
            if not extract_archive(archive_file, extract_dir):
                print(f"Skipping {archive_file.name} - extraction failed")
                failed_merges += 1
                continue
            
            # Find .ts file pairs
            pair_result = find_ts_pairs(extract_dir)
            
            if not pair_result:
                print(f"Skipping {archive_file.name} - no valid TRAIN/TEST pairs found")
                failed_merges += 1
                continue
            
            train_file, test_file, base_name = pair_result
            print(f"Found pair: {Path(train_file).name} + {Path(test_file).name}")
            
            # Generate output filename using the common base name
            output_filename = f"{base_name}.ts"
            output_file = output_dir / output_filename
            
            # Perform merge
            success = merge_timeseries(train_file, test_file, str(output_file))
            
            if success:
                print(f"Successfully merged: {output_filename}")
                successful_merges += 1
                
                # Optionally keep extracted files
                if keep_extracted:
                    extracted_dir = output_dir / f"{archive_file.stem}_extracted"
                    shutil.copytree(extract_dir, extracted_dir)
                    print(f"Extracted files saved to: {extracted_dir}")
                    
            else:
                print(f"Failed to merge files from {archive_file.name}")
                failed_merges += 1
    
    print(f"\n{'='*50}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Successful merges: {successful_merges}")
    print(f"Failed merges: {failed_merges}")
    print(f"Output directory: {output_dir.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Batch merge time series datasets from compressed files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_merge.py --data-dir ./compressed_datasets
  python batch_merge.py --data-dir ./compressed_datasets --output-dir ./my_results
  python batch_merge.py --data-dir ./data --keep-extracted
        """
    )
    
    parser.add_argument('--data-dir', required=True, 
                       help='Directory containing compressed files with .ts datasets')
    parser.add_argument('--output-dir', 
                       help='Directory to save merged datasets (default: data-dir/merged)')
    parser.add_argument('--keep-extracted', action='store_true',
                       help='Keep extracted files after processing')
    
    args = parser.parse_args()
    
    # Execute batch merge
    batch_merge_compressed_datasets(
        data_directory=args.data_dir,
        output_directory=args.output_dir,
        keep_extracted=args.keep_extracted
    )