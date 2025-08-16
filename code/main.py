import argparse
from process import process

def main():
    """
    Command-line interface for time series classification with optional random projection.

    This script serves as a front-end for the 'process' function defined in 'process.py'.
    It allows users to specify dataset path, model type, projection settings, 
    and evaluation parameters directly from the command line.
    """

    # Create an argument parser for command-line execution
    parser = argparse.ArgumentParser(
        description="Time series classification with random projection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Path to the .ts file containing the dataset
    parser.add_argument(
        "-p", "--path", 
        required=True,
        help="Path to the .ts file containing the dataset"
    )
    
    # Model type selection
    parser.add_argument(
        "-m", "--model",
        choices=['knn', 'ee', 'hivecotev1', 'hivecotev2'],
        default='knn',
        help="Type of model to use"
    )
    
    # Distance metric for k-NN
    parser.add_argument(
        "-d", "--dist",
        default='euclidean',
        help="Distance metric for KNN (euclidean, dtw, etc.)"
    )
    
    # Number of neighbors for k-NN
    parser.add_argument(
        "-n", "--k_neighbors",
        type=int,
        default=1,
        help="Number of neighbors for KNN"
    )
    
    # Whether to plot a learning curve or just do cross-validation
    parser.add_argument(
        "-l", "--lc",
        choices=['y', 'n'],
        default='y',
        help="Use learning curve (y) or not (n)"
    )
    
    # Number of points for the learning curve
    parser.add_argument(
        "--lc_splits",
        type=int,
        default=10,
        help="Number of points for the learning curve"
    )
    
    # Projection type
    parser.add_argument(
        "--proj",
        choices=['gaussian', 'sparse', 'no'],
        default='gaussian',
        help="Type of random projection"
    )
    
    # Epsilon parameter for sparse projection
    parser.add_argument(
        "--eps",
        type=float,
        default=0.2,
        help="Epsilon parameter for the distortion accepted through projection"
    )
    
    # Number of folds for cross-validation
    parser.add_argument(
        "--cv_splits",
        type=int,
        default=5,
        help="Number of folds for cross-validation"
    )
    
    # Global random state applied to all components
    parser.add_argument(
        "-r", "--rs",
        type=int,
        default=None,
        help="Global random state (applied to all components)"
    )
    
    # Random state for projection step
    parser.add_argument(
        "--rs1",
        type=int,
        default=12,
        help="Random state for projection"
    )
    
    # Random state for the classifier
    parser.add_argument(
        "--rs2",
        type=int,
        default=21,
        help="Random state for the classifier"
    )
    
    # Random state for StratifiedKFold
    parser.add_argument(
        "--rs3",
        type=int,
        default=42,
        help="Random state for StratifiedKFold"
    )
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Call the processing function with parsed arguments
    process(
        path=args.path,
        model=args.model,
        dist=args.dist,
        k_neighbors=args.k_neighbors,
        lc=args.lc,
        lc_splits=args.lc_splits,
        proj=args.proj,
        eps=args.eps,
        cv_splits=args.cv_splits,
        rs=args.rs,
        rs1=args.rs1,
        rs2=args.rs2,
        rs3=args.rs3
    )
    
    # Confirmation message
    print("Processing completed successfully!")

if __name__ == "__main__":
    main()