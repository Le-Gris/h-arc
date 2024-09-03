import json
import polars as pl
from argparse import ArgumentParser
from utils import grid2str
import os
from pathlib import Path

basepath = Path(__file__).parent.parent


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--submission_id", type=str, required=True)
    return parser.parse_args()


def kaggle_submision_to_csv(input_json, submission_id):
    dataframe = {
        "task_name": [],
        "test_number": [],
        "submission_id": [],
        "test_output_grid": [],
        "attempt_number": [],
    }
    for k, test_list in input_json.items():
        for i, test_submissions in enumerate(test_list):
            for attempt, grid in test_submissions.items():
                dataframe["task_name"].append(k + ".json")
                dataframe["test_number"].append(i + 1)
                dataframe["submission_id"].append(submission_id)
                dataframe["test_output_grid"].append(grid2str(grid))
                dataframe["attempt_number"].append(int(attempt.split("_")[-1]))

    df = pl.DataFrame(dataframe)
    output_path = os.path.join(
        basepath, "data", "kaggle_solutions", submission_id, "submission.csv"
    )
    df.write_csv(output_path)


if __name__ == "__main__":
    args = get_args()
    input_json_path = os.path.join(
        basepath, "data", "kaggle_solutions", args.submission_id, "submission.json"
    )
    with open(input_json_path, "r") as f:
        input_json = json.load(f)
    kaggle_submision_to_csv(input_json, args.submission_id)
