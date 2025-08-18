import os
import zipfile
from pathlib import Path
import argparse
import shutil

def extract_all_ts_files(input_directory, extracted_folder_name="extracted_ts"):
    """
    Extracts all .ts files from zip archives to a permanent folder,
    then collects and displays all .ts file paths.
    
    Args:
        input_directory (str): Directory containing zip files
        extracted_folder_name (str): Name of the subdirectory where .ts files will be extracted
    """
    
    input_path = Path(input_directory)
    
    if not input_path.exists():
        print(f"Error: Directory '{input_directory}' does not exist")
        return
    
    if not input_path.is_dir():
        print(f"Error: '{input_directory}' is not a directory")
        return
    
    # Create destination folder for .ts files
    extracted_path = input_path / extracted_folder_name
    
    # If folder already exists, ask for confirmation
    if extracted_path.exists():
        response = input(f"Directory '{extracted_path}' already exists. Do you want to delete it and start over? (y/N): ")
        if response.lower() in ['y', 'yes']:
            shutil.rmtree(extracted_path)
            print(f"Directory '{extracted_path}' deleted.")
        else:
            print("Using existing directory...")
    
    # Create extraction directory
    extracted_path.mkdir(parents=True, exist_ok=True)
    
    # Find all zip files
    zip_files = list(input_path.glob("*.zip"))
    
    if not zip_files:
        print(f"No zip files found in '{input_directory}'")
        return
    
    # Sort zip files by name
    zip_files.sort()
    
    print(f"Extracting {len(zip_files)} zip files...")
    print(f"Destination: {extracted_path}")
    print("="*80)
    
    total_ts_files = 0
    
    # Phase 1: Extract .ts files
    for i, zip_file in enumerate(zip_files, 1):
        print(f"[{i}/{len(zip_files)}] Processing: {zip_file.name}")
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Get list of files in the archive
                file_list = zip_ref.namelist()
                
                # Filter to keep only .ts files
                ts_files_in_zip = [f for f in file_list if f.endswith('.ts')]
                
                if not ts_files_in_zip:
                    print(f"  Warning: No .ts files found in {zip_file.name}")
                    continue
                
                print(f"  Success: Extracting {len(ts_files_in_zip)} .ts file(s):")
                
                # Extract each .ts file
                for ts_file in ts_files_in_zip:
                    # Extract the file
                    zip_ref.extract(ts_file, extracted_path)
                    
                    # Get only the filename (without path from zip)
                    ts_filename = Path(ts_file).name
                    print(f"    - {ts_filename}")
                    total_ts_files += 1
                    
                    # If file was in a subfolder of the zip, move it to root
                    extracted_file_path = extracted_path / ts_file
                    final_file_path = extracted_path / ts_filename
                    
                    if extracted_file_path != final_file_path:
                        # Move file to root of extraction directory
                        shutil.move(str(extracted_file_path), str(final_file_path))
                        
                        # Remove empty directories created
                        parent_dir = extracted_file_path.parent
                        try:
                            if parent_dir != extracted_path and not any(parent_dir.iterdir()):
                                parent_dir.rmdir()
                        except:
                            pass  # Ignore if we cannot remove directory
        
        except zipfile.BadZipFile:
            print(f"  Error: {zip_file.name} is not a valid zip file")
            continue
        except Exception as e:
            print(f"  Error extracting {zip_file.name}: {e}")
            continue
    
    print("="*80)
    print(f"Extraction completed: {total_ts_files} .ts files extracted")
    print()
    
    # Phase 2: Collect .ts file paths
    print("Collecting .ts file paths...")
    print("="*80)
    
    # Get all .ts files from extraction directory
    ts_files = list(extracted_path.glob("*.ts"))
    
    if not ts_files:
        print("No .ts files found in extraction directory")
        return
    
    # Sort files by name (this will automatically put TRAIN before TEST for each pair)
    ts_files.sort()
    
    # Convert to absolute paths
    ts_paths = [str(ts_file.absolute()) for ts_file in ts_files]
    
    # Format output with quotes
    output_text = ' '.join(f'"{path}"' for path in ts_paths)
    
    # Display results
    print(f".ts files found ({len(ts_files)} files):")
    for i, ts_file in enumerate(ts_files, 1):
        print(f"  {i:2d}. {ts_file.name}")
    
    print("\n" + "="*80)
    print("FORMATTED PATHS:")
    print("="*80)
    print(output_text)
    print("="*80)
    
    return output_text, extracted_path

def save_paths_to_file(paths_text, output_file):
    """Saves paths to a file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(paths_text)
        print(f"\nPaths saved to: {output_file}")
    except Exception as e:
        print(f"\nError saving file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract all .ts files from zip archives to a permanent folder and display their paths',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction (uses default folder name 'extracted_ts')
  python extract.py ./input
  
  # With custom folder name
  python extract.py ./input --folder my_ts_files
  
  # With path saving to file
  python extract.py ./input --save-paths ts_paths.txt
  
  # Complete example with custom folder and path saving
  python extract.py ./input --folder extracted_ts --save-paths paths.txt
        """
    )
    
    parser.add_argument('input_directory', 
                       help='Directory containing zip files')
    parser.add_argument('--folder', '-f', default='extracted_ts',
                       help='Name of subdirectory where .ts files will be extracted (default: extracted_ts)')
    parser.add_argument('--save-paths', '-s',
                       help='File to save extracted paths')
    
    args = parser.parse_args()
    
    # Execute extraction and path collection
    result = extract_all_ts_files(
        input_directory=args.input_directory,
        extracted_folder_name=args.folder
    )
    
    if result and args.save_paths:
        paths_text, extracted_path = result
        save_paths_to_file(paths_text, args.save_paths)