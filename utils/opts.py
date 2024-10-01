import argparse
import sys


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--make_dataset',
        action='store_true',
        help='Download and extract the dataset.'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the model.'
    )

    parser.add_argument(
        '--model',
        type=str,
        required='--train' in sys.argv,
        help='Model to be trained.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of epochs to train the model.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training.'
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=900,
        help='Number of frames to sample from each video.'
    )
    parser.add_argument(
        '--target_size',
        type=int,
        nargs=2,
        default=112,
        help='Target size for the video frames.'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=10,
        help='Number of classes in the dataset.'
    )

    opts = parser.parse_args()

    # Ensure that either --make_dataset or --train is provided
    if not opts.make_dataset and not opts.train:
        parser.print_help()
        print("\nError: You must specify either --make_dataset or --train.")
        sys.exit(1)

    return opts
