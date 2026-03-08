"""
Training script for the puzzle-piece shape classifier.

Run this script once (or whenever you want to retrain) to generate synthetic
puzzle-piece and non-puzzle-piece shapes, perform a train / validation / test
split, fit a Random Forest classifier, and save the resulting model to
``puzzle_piece_model.pkl``.

Usage::

    python train_puzzle_classifier.py [--n_samples N] [--seed S] [--model_path PATH]

The saved model is loaded automatically by ``ImageAligner`` at runtime.
"""

import argparse

from puzzle_piece_classifier import train_classifier, MODEL_PATH


def main():
    parser = argparse.ArgumentParser(
        description='Train the puzzle-piece shape classifier.')
    parser.add_argument(
        '--n_samples', type=int, default=2000,
        help='Total number of synthetic training samples (balanced). Default: 2000.')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility. Default: 42.')
    parser.add_argument(
        '--model_path', type=str, default=MODEL_PATH,
        help=f'Path to save the trained model. Default: {MODEL_PATH}')

    args = parser.parse_args()

    print("=== Puzzle-piece shape classifier training ===\n")
    clf = train_classifier(
        n_samples=args.n_samples,
        seed=args.seed,
        model_path=args.model_path,
        verbose=True,
    )
    print("\nTraining complete.")
    return clf


if __name__ == '__main__':
    main()
