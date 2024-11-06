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
        '--multimodal',
        type=bool,
        default=False,
        help='Training multimodal model.'
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
        default=32,
        help='Number of frames to sample from each video.'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=10,
        help='Number of classes in the dataset.'
    )
    parser.add_argument(
        '--resume',
        type=bool,
        default=False,
        help='Resume training from a checkpoint.'
    )
    parser.add_argument(
        '--path_to_model',
        type=str,
        help='Resume training from a checkpoint.'
    )
    parser.add_argument(
        '--path_to_optimizer',
        type=str,
        help='Resume training from a checkpoint.'
    )

    opts = parser.parse_args()

    # Ensure that either --make_dataset or --train is provided
    if not opts.make_dataset and not opts.train:
        parser.print_help()
        print("\nError: You must specify either --make_dataset or --train.")
        sys.exit(1)

    if opts.resume:
        if not opts.path_to_model or not opts.path_to_optimizer:
            parser.print_help()
            print(
                "\nError: You must specify --path_to_model and --path_to_optimizer when --resume is True.")
            sys.exit(1)

    return opts
