from results import results
import argparse

def reproduce():
    """
    Command-line interface to reproduce the results obtained through our experiments.

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
        required=True,
        help="Paths to the .ts files containing the dataset"
    )

    # Parse command-line arguments
    args = parser.parse_args()

    results(args.paths)

    # Confirmation message
    print("Processing completed successfully!")

if __name__ == "__main__":
    reproduce()