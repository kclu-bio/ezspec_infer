import os
import time
from easydict import EasyDict
import yaml
from logging import Logger
import logging
import pandas as pd
import numpy as np
import math
from typing import List, Tuple


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s::%(name)s::%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, "log.txt"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root="./logs", prefix="", tag=""):
    time_data_tag = time.strftime("%Y_%m_%d", time.localtime())
    hour_minute_tag = time.strftime("%H_%M_%S", time.localtime())

    if prefix != "":
        hour_minute_tag = prefix + "_" + hour_minute_tag
    if tag != "":
        hour_minute_tag = hour_minute_tag + "_" + tag
    log_dir = os.path.join(root, time_data_tag, hour_minute_tag)
    # log_dir = os.path.join(log_dir, running_tag)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def load_config(path):
    with open(path, "r") as f:
        return EasyDict(yaml.safe_load(f))


def get_ranks_percentiles(
    holdout_df: pd.DataFrame,
    model_out: pd.DataFrame,
    probs_col: str = "score",
    ascending: bool = False,
) -> Tuple[List, List, pd.DataFrame]:
    """
    Collects the dataframes from the model output directory and the holdout dataframe.

    Args:
        holdout_df (pd.DataFrame): Dataframe containing the holdout data
        model_output (pd.DataFrame): Dataframe of model output predictions
        probs_col (str): Column name in model_out containing the prediction probabilities
        ascending (bool): Whether to sort the predictions in ascending order (when higher is better)

    Returns:
        Tuple: Lists of dataframes containing the model output and the holdout data,
                and the y-true labels and y-predictions
    """

    ## Get output df and holdout df
    ## Go through each row in holdout df, extract that dataset's predictions from output df
    ## Make sure there's only one " correct" prediction per dataset

    ## Sort the rows in that dataset by score
    ## Get the rank, percentile of the true enzyme
    ## Store the ranks percentiles in a list
    ## Return the two lists and the output dataframes

    df = model_out
    df["eval_set"] = df["pdb_id"].str.split("_").str[0:3].str.join("_")

    y_true = df.Label.tolist()
    try:
        y_pred = df[probs_col].tolist()
    except:
        print(f"column {probs_col} not found! Available columns: {df.columns}")
    ranks, percentiles = [], []

    for i, row in holdout_df.iterrows():
        rxnid = row["sample_id"]

        try:
            dataset = df[df["eval_set"] == rxnid]
        except KeyError:
            print(f"Dataset {rxnid} not found in model output. Skipping...")
            continue

        if dataset.empty:
            print(f"No predictions found for {rxnid}. Skipping...")
            continue

        ## Make sure there's only one "correct" prediction
        if dataset.Label.sum() != 1:
            print(f"Multiple or no correct predictions for {rxnid}. Skipping...")

            ## Remove this dataset from the output df
            df = df[df["eval_set"] != rxnid]
            continue

        ## Sort predictions by score
        dataset_sorted = dataset.sort_values(
            by=probs_col, ascending=ascending
        ).reset_index(drop=True)
        true_index = dataset_sorted[dataset_sorted["Label"] == 1].index[0]

        total = len(dataset_sorted)
        percentile = 100 * (true_index + 1) / total

        y_true.extend(dataset_sorted["Label"].tolist())
        y_pred.extend(dataset_sorted[probs_col].tolist())
        ranks.append(true_index + 1)
        percentiles.append(percentile)

    return y_true, y_pred, ranks, percentiles, df


def bedroc_score(y_true, y_pred, decreasing=True, alpha=20.0):
    """BEDROC metric implemented according to Truchon and Bayley.

    The Boltzmann Enhanced Descrimination of the Receiver Operator
    Characteristic (BEDROC) score is a modification of the Receiver Operator
    Characteristic (ROC) score that allows for a factor of *early recognition*.

    References:
        The original paper by Truchon et al. is located at `10.1021/ci600426e
        <http://dx.doi.org/10.1021/ci600426e>`_.

    Args:
        y_true (array_like):
            Binary class labels. 1 for positive class, 0 otherwise.
        y_pred (array_like):
            Prediction values.
        decreasing (bool):
            True if high values of ``y_pred`` correlates to positive class.
        alpha (float):
            Early recognition parameter.

    Returns:
        float:
            Value in interval [0, 1] indicating degree to which the predictive
            technique employed detects (early) the positive class.
    """

    assert len(y_true) == len(
        y_pred
    ), "The number of scores must be equal to the number of labels"

    big_n = len(y_true)
    n = sum(y_true == 1)

    if decreasing:
        order = np.argsort(-y_pred)
    else:
        order = np.argsort(y_pred)

    m_rank = (y_true[order] == 1).nonzero()[0]

    s = np.sum(np.exp(-alpha * m_rank / big_n))

    r_a = n / big_n

    rand_sum = r_a * (1 - np.exp(-alpha)) / (np.exp(alpha / big_n) - 1)

    fac = (
        r_a
        * np.sinh(alpha / 2)
        / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * r_a))
    )

    cte = 1 / (1 - np.exp(alpha * (1 - r_a)))

    return s * fac / rand_sum + cte


def enrichment_score(y_true, y_pred, decreasing=True, chi=0.1):
    """Enrichment Factor metric

        How many more actives we find within a defined
        “early recognition” fraction of the ordered
        list relative to a random distribution


              n_actives_in_sampled_set / n_sampled_set      n_actives_in_sampled_set
        EF =  ----------------------------------------  = ----------------------------
                   total_actives / total_ligands               chi * total_actives

    Args:
        y_true (array_like):
            Binary class labels. 1 for positive class, 0 otherwise.
        y_pred (array_like):
            Prediction values.
        decreasing (bool):
              True if high values of ``y_pred`` correlates to positive class.

    Returns:
        float:
            Value in interval [0, tau]
                tau = 1/chi if chi >= n/N
                tau = N/n if chi < n/N
    """
    big_n = len(y_true)
    n = sum(y_true == 1)

    if decreasing:
        order = np.argsort(-y_pred)
    else:
        order = np.argsort(y_pred)

    k = math.floor(chi * big_n)

    num_in_topk = (y_true[order] == 1)[:k].sum()

    ef = (num_in_topk) / (chi * n)

    return ef


def compute_topkpercent_accuracy(percentiles, k=10):
    """
    Compute the top-k percent accuracy.

    Args:
        percentiles (list): List of percentiles for each sample.
        k (int): The top k percent to consider.

    Returns:
        float: The top-k percent accuracy.
    """

    passing = [p for p in percentiles if p <= k]
    top_k_accuracy = len(passing) / len(percentiles)
    return top_k_accuracy
