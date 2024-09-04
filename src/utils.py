from datetime import datetime as dt
import polars as pl
import hashlib


def parse_mixed_datetime(datetime_str):
    formats = ["%m/%d/%Y, %I:%M:%S %p", "%m/%d/%Y, %H:%M:%S"]
    for fmt in formats:
        try:
            return dt.strptime(datetime_str, fmt)
        except ValueError:
            continue
    return None


def get_summary(df, verbose=False):
    """Get ARC summary data frame by filtering out traces where last action is invalid"""
    # list of final actions
    final_actions = [
        "no_last_description",
        "write_description",
        "write_last_description",
    ]
    df = df.with_columns(
        attempt_number=pl.when(pl.col("attempt_number") == 4)
        .then(3)
        .otherwise(pl.col("attempt_number"))
    )
    complete_task_joint_ids = (
        df.select(
            pl.all()
            .top_k_by(["attempt_number", "num_actions"], k=1)
            .over(["joint_id_task"], mapping_strategy="explode")
        )
        .filter(pl.col("action").is_in(final_actions))
        .select("joint_id_task")
    )
    df_summary = df.join(complete_task_joint_ids, on="joint_id_task").select(
        pl.all()
        .top_k_by("num_actions", k=1)
        .over(["joint_id_task", "attempt_number"], mapping_strategy="explode")
    )
    if verbose:
        print(
            f"Filtered out {df.n_unique('joint_id_task') - df_summary.n_unique('joint_id_task')}/{df.n_unique('joint_id_task')} incomplete participant task attempts"
        )
    df_summary = df_summary[
        [
            "hashed_id",
            "task_name",
            "joint_id_task",
            "worker_id",
            "task_number",
            "attempt_number",
            "action",
            "solved",
            "test_output_grid",
            "first_written_solution",
            "last_written_solution",
            "num_actions",
            "exp_name",
            "task_type",
        ]
    ]
    return df_summary


def get_errors(df):
    """Take ARC summary data frame and filter for all incorrect attempts"""
    df_errors = df.filter(pl.col("solved") == False)

    # get frequency of errors
    df_errors = df_errors.group_by(["task_name", "test_output_grid"]).agg(
        pl.count().alias("count"),
        pl.first("task_type").alias("task_type"),
    )

    # get hashed output grid
    df_errors = df_errors.with_columns(
        pl.col("test_output_grid")
        .map_elements(
            lambda x: hashlib.md5(x.encode()).hexdigest(), return_dtype=pl.Utf8
        )
        .alias("hashed_output_grid")
    )

    return df_errors


def grid2str(grid):
    """Converts an ARC grid in numpy form to a string representation"""
    grid_str = "|"
    for row in grid:
        for num in row:
            grid_str += str(num)
        grid_str += "|"
    return grid_str


def include_incomplete(df_summary, df_incomplete, verbose=False):
    df_summary = df_summary.drop("condition")
    df_incomplete_summary = get_summary(df_incomplete, verbose=verbose)
    df_incomplete_summary = df_incomplete_summary.select(df_summary.columns)
    df_summary = df_summary.with_columns(pl.lit(False).alias("incomplete"))
    df_incomplete_summary = df_incomplete_summary.with_columns(
        pl.lit(True).alias("incomplete")
    )
    df_summary = df_summary.vstack(df_incomplete_summary)
    if verbose:
        print(
            f"Included {df_incomplete_summary.n_unique('joint_id_task')}/{df_incomplete.n_unique('joint_id_task')} incomplete participant task attempts"
        )
    return df_summary, df_incomplete_summary


def md5(grid):
    """Converts a string representation of a grid to an md5 hash for indexing"""
    return hashlib.md5(bytes(grid, encoding="utf-8")).hexdigest()
