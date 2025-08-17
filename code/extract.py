import os
from pathlib import Path
import argparse

def extract_file_paths(directory, output_file=None, file_extension=None):
    """
    Extract all file paths from a directory and display them separated by spaces.
    
    Args:
        directory (str): Directory to scan for files
        output_file (str, optional): File to save the output to
        file_extension (str, optional): Filter by file extension (e.g., '.ts')
    """
    
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Error: Directory '{directory}' does not exist")
        return
    
    if not directory_path.is_dir():
        print(f"Error: '{directory}' is not a directory")
        return
    
    # Get all files in directory
    if file_extension:
        files = list(directory_path.glob(f"*{file_extension}"))
    else:
        files = [f for f in directory_path.iterdir() if f.is_file()]
    
    if not files:
        print(f"No files found in directory '{directory}'")
        if file_extension:
            print(f"(Looking for files with extension: {file_extension})")
        return
    
    # Sort files by name for consistent output
    files.sort()
    
    # Convert paths to strings and join with spaces
    file_paths = [str(f.absolute()) for f in files]
    output_text = ' '.join(file_paths)
    
    # Display results
    print(f"Found {len(files)} files in '{directory}':")
    print("\n" + "="*80)
    print(output_text)
    print("="*80 + "\n")
    
    # Save to file if specified
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output_text)
            print(f"File paths saved to: {output_file}")
        except Exception as e:
            print(f"Error saving to file: {e}")
    
    return output_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract file paths from a directory and display them separated by spaces',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_paths.py --directory ./merged
  python extract_paths.py --directory ./merged --extension .ts
  python extract_paths.py --directory ./merged --output paths.txt
  python extract_paths.py --directory ./merged --extension .ts --output ts_paths.txt
        """
    )
    
    parser.add_argument('--directory', '-d', required=True,
                       help='Directory containing the files')
    parser.add_argument('--extension', '-e',
                       help='Filter by file extension (e.g., .ts, .txt)')
    parser.add_argument('--output', '-o',
                       help='Output file to save the paths')
    
    args = parser.parse_args()
    
    extract_file_paths(
        directory=args.directory,
        output_file=args.output,
        file_extension=args.extension
    )