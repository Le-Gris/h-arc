import pymc
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import scipy.stats as stats
from matplotlib.patches import Patch
from src.utils import *
import argparse as ap
import os
import json
from pathlib import Path
import cloudpickle as cpkl
from src.bayesian_IRT import bayes_irt

basepath = Path(__file__).parent.parent


def get_args():
    parser = ap.ArgumentParser(description="Generate IRT model results")
    parser.add_argument("--model", type=str, help="path to model file")
    parser.add_argument("--n_samples", type=int, default=5000, help="number of samples")
    parser.add_argument(
        "--n_burn", type=int, default=1000, help="number of burn in samples"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for random number generator"
    )
    parser.add_argument(
        "--impute", type=bool, default=False, help="impute missing data"
    )
    return parser.parse_args()


def load_data():
    # load data
    df_summary = pl.read_csv(os.path.join(basepath, "data", "clean_summary_data.csv"))
    df_summary_incomplete = pl.read_csv(
        os.path.join(basepath, "data", "clean_summary_data_incomplete.csv")
    )

    # combine data
    df_summary = pl.concat([df_summary, df_summary_incomplete])

    # filter data
    columns = [
        "hashed_id",
        "task_name",
        "attempt_number",
        "solved",
        "task_type",
        "complete",
    ]
    df = (
        df_summary.select(columns)
        .with_columns(
            pl.col("attempt_number").cast(pl.Int32),
            pl.col("solved").cast(pl.Int32),
        )
        .to_pandas()
    )

    # reshape for bernouilli
    df = df.pivot_table(
        values="solved",
        index=["hashed_id", "task_name", "task_type", "complete"],
        columns="attempt_number",
        aggfunc="first",
    ).reset_index()
    df.columns.name = None
    df = df.rename(columns={1: "1-shot", 2: "2-shots", 3: "3-shots"})
    df = df.fillna(1)
    return df


def get_incomplete_tasks(df, ordered_tasks_list):
    incomplete = df.groupby("hashed_id").size()
    incomplete = incomplete.reset_index()
    incomplete = incomplete.rename(columns={0: "n_complete"})
    incomplete = incomplete[incomplete["n_complete"] < 5]
    repeats = 5 - incomplete["n_complete"].to_numpy().flatten()

    tasks = []
    for h, m, r in zip(incomplete["hashed_id"], incomplete["n_complete"], repeats):
        # Get completed tasks for this participant
        completed_tasks = df[df["hashed_id"] == h]["task_name"].to_numpy()
        # Find indices of completed tasks in ordered list
        completed_indices = np.where(np.isin(ordered_tasks_list, completed_tasks))[0]

        if len(completed_indices) == 0:
            # If no tasks completed, start from beginning
            next_tasks = []
            idx = 0
            while len(next_tasks) < r:
                if ordered_tasks_list[idx] not in completed_tasks:
                    next_tasks.append(ordered_tasks_list[idx])
                idx = (idx + 1) % len(ordered_tasks_list)
        else:
            # Start from the highest completed index
            next_tasks = []
            idx = (completed_indices.max() + 1) % len(ordered_tasks_list)
            while len(next_tasks) < r:
                if ordered_tasks_list[idx] not in completed_tasks:
                    next_tasks.append(ordered_tasks_list[idx])
                idx = (idx + 1) % len(ordered_tasks_list)

        tasks.extend(next_tasks)

    hashed_ids = incomplete["hashed_id"].repeat(repeats)
    return hashed_ids, np.array(tasks)


def load_ordered_tasks():
    df_training_tasks_ordered = json.load(
        open(os.path.join(basepath, "data", "ARC_training_tasks_ordered.json"))
    )
    df_eval_tasks_ordered = json.load(
        open(os.path.join(basepath, "data", "ARC_evaluation_tasks_ordered.json"))
    )
    return df_training_tasks_ordered, df_eval_tasks_ordered


def fill_na(df, hashed_ids, tasks):
    nan_rows = pd.DataFrame(
        {
            "hashed_id": hashed_ids,
            "task_name": tasks,
            "task_type": "training",
            "complete": False,
            "1-shot": np.nan,
            "2-shots": np.nan,
            "3-shots": np.nan,
        }
    )
    df = pd.concat([df, nan_rows])
    return df


def include_missing_data(df):
    # separate training and evaluation data
    df_training = df[df["task_type"] == "training"]
    df_eval = df[df["task_type"] == "evaluation"]
    # load ordered tasks
    df_training_tasks_ordered, df_eval_tasks_ordered = load_ordered_tasks()
    # get incomplete tasks
    hashed_ids_training, tasks_training = get_incomplete_tasks(
        df_training, np.array(df_training_tasks_ordered)
    )
    hashed_ids_eval, tasks_eval = get_incomplete_tasks(
        df_eval, np.array(df_eval_tasks_ordered)
    )
    # fill missing data
    df_training = fill_na(df_training, hashed_ids_training, tasks_training)
    df_eval = fill_na(df_eval, hashed_ids_eval, tasks_eval)
    # combine data
    df = pd.concat([df_training, df_eval])
    return df


if __name__ == "__main__":
    args = get_args()
    # load data
    df = load_data()
    if args.impute:
        df = include_missing_data(df)
    # run IRT model
    model, trace = bayes_irt(
        df, n_samples=args.n_samples, tune=args.n_burn, seed=args.seed
    )
    # save model
    imputed = "_imputed" if args.impute else ""
    with open(
        os.path.join(basepath, "models", f"bayes_IRT_model{imputed}_{args.seed}.pkl"),
        "wb",
    ) as f:
        cpkl.dump((model, trace), f)
