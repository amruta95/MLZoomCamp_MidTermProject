import argparse
import json
import os

from sklearn.metrics import f1_score, precision_score, recall_score


def main():
    # Read true and prediction data file names from disk
    with open(args.true, "r") as f:
        true_data = json.load(f)
    with open(args.predictions, "r") as f:
        predictions = json.load(f)

    # Get all true ids and labels
    true_ids = [t["id"] for t in true_data]
    true_labels = [t["label"] for t in true_data]

    # Retrieve predicted labels in the same order as the true data
    y_pred = [int(predictions[i]) for i in true_ids]

    # Calculate actual score
    print("Precision:", precision_score(true_labels, y_pred))
    print("Recall:", recall_score(true_labels, y_pred))
    print("F1:", f1_score(true_labels, y_pred))


if __name__ == "__main__":
    # Add cli parameters
    parser = argparse.ArgumentParser("Script to evaluate the predictions on a test set.")

    parser.add_argument(
        "--true",
        "-t",
        required=True,
        help="Path to the test data directory.",
        metavar="TEST_DATA")
    parser.add_argument(
        "--predictions",
        "-p",
        required=True,
        help="Path to the predictions file in the specified format.",
        metavar="PREDICTIONS")

    args = parser.parse_args()

    main()

    print("Done.")
