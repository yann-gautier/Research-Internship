from results import results
import argparse

def reproduce():
    """
    Command-line interface to reproduce the results obtained through our experiments.
    
    Note :
    One should examine the entry format
    """
    
    # Create an argument parser for command-line execution
    parser = argparse.ArgumentParser(
        description="Results reproducibility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Paths to the .ts files containing the dataset
    parser.add_argument(
        "-p", "--paths", 
        nargs='+',
        required=True,
        help="Paths to the .ts files containing the datasets"
    )
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # args.paths
    results(args.paths)
    
    # Confirmation message
    print("Processing completed successfully!")

if __name__ == "__main__":
    reproduce()