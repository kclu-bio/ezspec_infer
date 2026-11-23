import pandas as pd
import numpy as np
import math
from pathlib import Path
from typing import List, Tuple, Dict
from utils import *
import argparse


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model-out", type=str, required=True)
    parser.add_argument("-holdout", type=str, required=True)
    return parser.parse_args()


def main():
    args = argparser()
    holdout_df = pd.read_csv(args.holdout)

    eval_results = pd.read_csv(args.model_out)

    y_true, y_pred, ranks, percentiles, preddf = get_ranks_percentiles(
        holdout_df, eval_results
    )

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute BEDROC and EF
    bedroc = bedroc_score(y_true, y_pred)
    ef = enrichment_score(y_true, y_pred)

    ## Compute top-k accuracies
    top10 = compute_topkpercent_accuracy(percentiles, k=10)
    top20 = compute_topkpercent_accuracy(percentiles, k=20)
    top50 = compute_topkpercent_accuracy(percentiles, k=50)

    # Mean Reciprocal Rank (MRR)
    reciprocal_ranks = [1.0 / r for r in ranks]
    mrr = float(np.mean(reciprocal_ranks))

    print(f"BEDROC: {bedroc}")
    print(f"Enrichment Factor: {ef}")
    print(f"Top 10% Accuracy: {top10}")
    print(f"Top 20% Accuracy: {top20}")
    print(f"Top 50% Accuracy: {top50}")
    print(f"Mean Rank: {np.mean(ranks)}")
    print(f"Mean Reciprocal Rank: {mrr}")
    print(f"Mean Percentile: {np.mean(percentiles)}")


if __name__ == "__main__":
    main()
