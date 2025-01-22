import argparse as ap
import os
import cloudpickle as cpkl
import arviz as az
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import numpy as np
import seaborn as sns

basepath = Path(__file__).parent.parent

LABEL_SIZE = 20
TITLE_SIZE = 24
TICK_SIZE = 18
LEGEND_SIZE = 18

plt.rcParams["text.usetex"] = True


def get_args():
    parser = ap.ArgumentParser(description="Generate IRT model plots and analysis")
    parser.add_argument("--model_path", type=str, help="path to model")
    parser.add_argument("--verbose", action="store_true", help="print verbose output")
    return parser.parse_args()


def load_model(model_path):
    with open(os.path.join(basepath, model_path), "rb") as f:
        model, trace = cpkl.load(f)
    return model, trace


def plot_trace(trace, model_name):
    ax = az.plot_trace(
        trace,
        var_names=[
            "sigma_alpha",
            "sigma_beta",
            "epsilon_one",
            "epsilon_delta",
            "epsilon_two",
        ],
    )
    fig = ax.ravel()[0].figure
    plt.tight_layout()
    fig.savefig(
        os.path.join(basepath, "figures", f"{model_name}_trace_plot.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def get_participant_success_rate():
    # load data
    df_summary = pl.read_csv(os.path.join(basepath, "data", "clean_summary_data.csv"))
    df_summary_incomplete = pl.read_csv(
        os.path.join(basepath, "data", "clean_summary_data_incomplete.csv")
    )
    df_summary = pl.concat([df_summary, df_summary_incomplete])

    # filter data
    participants_max_tasks = df_summary.group_by("hashed_id").agg(
        pl.max("task_number"), pl.first("task_type")
    )
    participants_max_tasks = participants_max_tasks.rename(
        {"task_number": "max_task_number"}
    )
    df_summary_ = df_summary.join(participants_max_tasks, on="hashed_id")
    participant_success = df_summary_.group_by("hashed_id").agg(
        pl.sum("solved") / pl.max("max_task_number"), pl.first("complete")
    )
    return participant_success


def generate_irt_dataframes(trace):
    # Load task success rate
    mean_task_acc_three_shot = pl.read_csv(
        os.path.join(basepath, "data", "mean_task_acc_three_attempts.csv")
    )
    # Extract necessary data from the trace
    alpha_mean = trace.posterior["alpha"].mean(dim=["chain", "draw"])
    alpha_hdi = az.hdi(trace.posterior["alpha"], hdi_prob=0.94)
    participants = trace.posterior["alpha"].coords["participants"].values
    participant_success = get_participant_success_rate()
    participant_success_values = (
        participant_success.select("hashed_id", "solved")
        .to_pandas()
        .set_index("hashed_id")
        .loc[participants]
        .values.flatten()
    )
    complete = (
        participant_success.select("hashed_id", "complete")
        .to_pandas()
        .set_index("hashed_id")
        .loc[participants]
        .values.flatten()
    )

    beta_mean = trace.posterior["beta"].mean(dim=["chain", "draw"])
    beta_hdi = az.hdi(trace.posterior["beta"], hdi_prob=0.94)
    tasks = trace.posterior["beta"].coords["tasks"].values

    # order success rate by tasks
    task_success_values = (
        mean_task_acc_three_shot.select("task_name", "mean_solved")
        .to_pandas()
        .set_index("task_name")
        .loc[tasks]
        .values.flatten()
    )

    task_type_values = (
        mean_task_acc_three_shot.select("task_name", "task_type")
        .to_pandas()
        .set_index("task_name")
        .loc[tasks]
        .values.flatten()
    )
    epsilon_one_mean = trace.posterior["epsilon_one"].mean(dim=["chain", "draw"])
    epsilon_two_mean = trace.posterior["epsilon_two"].mean(dim=["chain", "draw"])
    epsilon_one_hdi = az.hdi(trace.posterior["epsilon_one"], hdi_prob=0.94)
    epsilon_two_hdi = az.hdi(trace.posterior["epsilon_two"], hdi_prob=0.94)

    epsilon_one = trace.posterior["epsilon_one"].values.flatten()
    epsilon_two = trace.posterior["epsilon_two"].values.flatten()

    # Create DataFrame for participant abilities
    plot_df_ability = pl.DataFrame(
        {
            "hashed_id": participants,
            "ability_mean": alpha_mean.values,
            "success_rate": participant_success_values,
            "ability_hdi_lower": alpha_hdi.sel(hdi="lower").alpha.values,
            "ability_hdi_upper": alpha_hdi.sel(hdi="higher").alpha.values,
            "complete": complete,
        }
    )

    # Create DataFrame for task difficulties
    plot_df_difficulty = pl.DataFrame(
        {
            "tasks": tasks,
            "task_type": task_type_values,
            "success_rate": task_success_values,
            "diff_mean": beta_mean.values,
            "diff_hdi_lower": beta_hdi.sel(hdi="lower").beta.values,
            "diff_hdi_upper": beta_hdi.sel(hdi="higher").beta.values,
        }
    )

    # Create DataFrame for epsilon values
    plot_df_epsilon = pl.DataFrame(
        {
            "epsilon": list(epsilon_one) + list(epsilon_two),
            "attempt": [r"$\epsilon_1$"] * len(epsilon_one)
            + [r"$\epsilon_2$"] * len(epsilon_two),
        }
    )

    # create alternative dataframe for epsilon
    plot_df_epsilon_alt = pl.DataFrame(
        {
            "attempt": [1, 2],
            "epsilon_mean": [
                epsilon_one_mean.values.item(),
                epsilon_two_mean.values.item(),
            ],
            "epsilon_hdi_lower": [
                epsilon_one_hdi.sel(hdi="lower").epsilon_one.values.item(),
                epsilon_two_hdi.sel(hdi="lower").epsilon_two.values.item(),
            ],
            "epsilon_hdi_upper": [
                epsilon_one_hdi.sel(hdi="higher").epsilon_one.values.item(),
                epsilon_two_hdi.sel(hdi="higher").epsilon_two.values.item(),
            ],
        }
    )

    return plot_df_ability, plot_df_difficulty, plot_df_epsilon, plot_df_epsilon_alt


def plot_irt_parameters(df_epsilon, plot_df_ability, plot_df_difficulty, model_name):
    # Create figure and gridspec for the main layout
    fig = plt.figure(figsize=(20, 12))
    gs_main = plt.GridSpec(
        2, 2, width_ratios=[3, 5], height_ratios=[1, 1], hspace=0.25, wspace=0.175
    )

    eps_palette = sns.color_palette("Paired", 2)
    others_palette = sns.color_palette("tab10")

    # Create subgridspecs for the right plots
    gs_ability = gs_main[0, 1].subgridspec(1, 2, width_ratios=[4, 1], wspace=0.01)
    gs_diff = gs_main[1, 1].subgridspec(1, 2, width_ratios=[4, 1], wspace=0.01)

    # Left plot (epsilon distributions)
    ax_eps = fig.add_subplot(gs_main[0, 0])

    sns.histplot(
        df_epsilon,
        x="epsilon",
        hue="attempt",
        bins=50,
        alpha=0.8,
        palette=eps_palette,
        stat="density",
        kde=True,
        ax=ax_eps,
        legend=True,
    )
    ax_eps.set_title("(a) Feedback effect", fontsize=TITLE_SIZE)
    ax_eps.set_xlabel("Parameter estimates", fontsize=LABEL_SIZE)
    ax_eps.set_ylabel("Count", fontsize=LABEL_SIZE)
    ax_eps.set_xticklabels(ax_eps.get_xticks(), fontsize=TICK_SIZE)
    ax_eps.set_yticklabels(ax_eps.get_yticks(), fontsize=TICK_SIZE)
    handles = ax_eps.get_legend().legend_handles
    labels = [r"$\gamma_1$", r"$\gamma_2$"]
    ax_eps.legend(handles, labels, fontsize=LEGEND_SIZE)
    ax_eps.grid(True, alpha=0.1)

    # Top right plots (ability)
    ax_ability_main = fig.add_subplot(gs_ability[0, 0])
    ax_ability_kde = fig.add_subplot(gs_ability[0, 1])

    # Jittered success rates for scatter plots
    jittered_success_rate_ability = plot_df_ability.select(
        "success_rate"
    ).to_numpy().flatten() + np.random.normal(0, 0.01, len(plot_df_ability))

    # Create ability scatter plot
    ax_ability_main.scatter(
        jittered_success_rate_ability,
        plot_df_ability.select("ability_mean").to_numpy().flatten(),
        s=20,
        alpha=0.075,
        color="black",
        zorder=2,
    )

    ax_ability_main.set_title("(b) Participant ability", fontsize=TITLE_SIZE)
    ax_ability_main.set_xlabel("Mean participant accuracy", fontsize=LABEL_SIZE)
    ax_ability_main.set_xticklabels([])
    ax_ability_main.set_ylabel(r"$\alpha$", fontsize=LABEL_SIZE)
    ax_ability_main.set_yticklabels(ax_ability_main.get_yticks(), fontsize=TICK_SIZE)
    ax_ability_main.grid(True, alpha=0.1)
    ax_ability_main.axhline(0, color="black", linestyle="--", alpha=0.2)

    # Create ability KDE
    sns.kdeplot(
        plot_df_ability,
        y="ability_mean",
        ax=ax_ability_kde,
        fill=True,
        color=others_palette[4],
        alpha=0.1,
    )
    ax_ability_kde.set_yticklabels([])
    ax_ability_kde.set_xticklabels([])
    ax_ability_kde.set_xlabel("")
    ax_ability_kde.set_ylabel("")
    ax_ability_kde.spines["left"].set_visible(False)
    ax_ability_kde.spines["top"].set_visible(False)
    ax_ability_kde.spines["right"].set_visible(False)
    ax_ability_kde.spines["bottom"].set_visible(False)
    ax_ability_kde.tick_params(axis="y", left=False)
    ax_ability_kde.tick_params(axis="x", bottom=False)
    ax_ability_kde.set_ylim(ax_ability_main.get_ylim())

    # Bottom right plots (difficulty)
    ax_diff_main = fig.add_subplot(gs_diff[0, 0])
    ax_diff_kde = fig.add_subplot(gs_diff[0, 1])

    # Jittered success rates for scatter plots
    jittered_success_rate_difficulty = plot_df_difficulty.select(
        "success_rate"
    ).to_numpy().flatten() + np.random.normal(0, 0.01, len(plot_df_difficulty))

    # Create difficulty scatter plot
    ax_diff_main.scatter(
        jittered_success_rate_difficulty,
        plot_df_difficulty.select("diff_mean").to_numpy().flatten(),
        s=20,
        alpha=0.2,
        color="black",
        zorder=2,
    )
    ax_diff_main.errorbar(
        jittered_success_rate_difficulty,
        plot_df_difficulty["diff_mean"],
        yerr=[
            plot_df_difficulty["diff_mean"] - plot_df_difficulty["diff_hdi_lower"],
            plot_df_difficulty["diff_hdi_upper"] - plot_df_difficulty["diff_mean"],
        ],
        fmt="none",
        ecolor="gray",
        alpha=0.1,
        zorder=1,
    )

    # add mean difficulty for training and eval tasks
    mean_diff_training = (
        plot_df_difficulty.filter(pl.col("task_type") == "training")
        .select(pl.mean("diff_mean"))
        .item()
    )
    mean_diff_eval = (
        plot_df_difficulty.filter(pl.col("task_type") == "evaluation")
        .select(pl.mean("diff_mean"))
        .item()
    )

    ax_diff_main.axhline(
        mean_diff_training, color=others_palette[0], linestyle="--", alpha=0.5
    )
    ax_diff_main.axhline(
        mean_diff_eval, color=others_palette[1], linestyle="--", alpha=0.5
    )

    ax_diff_main.set_title("(c) Task difficulty", fontsize=TITLE_SIZE)
    ax_diff_main.set_xlabel("Mean task accuracy", fontsize=LABEL_SIZE)
    ax_diff_main.set_ylabel(r"$\beta$", fontsize=LABEL_SIZE)
    ax_diff_main.set_xticklabels(ax_diff_main.get_xticklabels(), fontsize=TICK_SIZE)
    ax_diff_main.set_yticklabels(ax_diff_main.get_yticklabels(), fontsize=TICK_SIZE)
    ax_diff_main.grid(True, alpha=0.1)
    ax_diff_main.axhline(0, color="black", linestyle="--", alpha=0.2)

    # create legend for kde
    legend_elements = [
        Patch(edgecolor=others_palette[0], label="Training set", fill=False),
        Patch(edgecolor=others_palette[1], label="Evaluation set", fill=False),
    ]

    # Add legend with custom handles
    ax_diff_main.legend(handles=legend_elements, loc="upper right")

    # Create difficulty KDE
    sns.kdeplot(
        plot_df_difficulty.filter(pl.col("task_type") == "training"),
        y="diff_mean",
        ax=ax_diff_kde,
        fill=True,
        color=others_palette[0],
        alpha=0.1,
    )
    sns.kdeplot(
        plot_df_difficulty.filter(pl.col("task_type") == "evaluation"),
        y="diff_mean",
        ax=ax_diff_kde,
        fill=True,
        color=others_palette[1],
        alpha=0.1,
    )
    ax_diff_kde.set_yticklabels([])
    ax_diff_kde.set_xticklabels([])
    ax_diff_kde.set_xlabel("")
    ax_diff_kde.set_ylabel("")
    ax_diff_kde.spines["left"].set_visible(False)
    ax_diff_kde.spines["top"].set_visible(False)
    ax_diff_kde.spines["right"].set_visible(False)
    ax_diff_kde.spines["bottom"].set_visible(False)
    ax_diff_kde.tick_params(axis="y", left=False)
    ax_diff_kde.tick_params(axis="x", bottom=False)
    ax_diff_kde.set_ylim(ax_diff_main.get_ylim())

    # make sure ability and diff share the same xlim
    ax_ability_main.set_xlim(ax_diff_main.get_xlim())

    # plt.tight_layout()
    plt.savefig(
        os.path.join(basepath, "figures", f"{model_name}_irt_parameters.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def forest_plots(plot_df_ability, plot_df_difficulty, model_name):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))

    # Ability Forest Plot
    ability_sorted = plot_df_ability.sort("ability_mean").with_row_index("idx")
    ability_sorted_complete = ability_sorted.filter(pl.col("complete"))

    # plot abiltiies for complete participants
    ax1.errorbar(
        x=ability_sorted_complete.select("ability_mean").to_numpy().flatten(),
        y=ability_sorted_complete.select("idx").to_numpy().flatten(),
        xerr=[
            ability_sorted_complete.select("ability_mean").to_numpy().flatten()
            - ability_sorted_complete.select("ability_hdi_lower").to_numpy().flatten(),
            ability_sorted_complete.select("ability_hdi_upper").to_numpy().flatten()
            - ability_sorted_complete.select("ability_mean").to_numpy().flatten(),
        ],
        fmt="o",
        color="black",
        markersize=1,
        elinewidth=0.5,
        capsize=0,
        alpha=0.1,
        label="Complete",
    )
    # add ability for incomplete participants with color
    ability_sorted_incomplete = ability_sorted.filter(~pl.col("complete")).sort(
        "ability_mean"
    )
    ax1.errorbar(
        x=ability_sorted_incomplete.select("ability_mean").to_numpy().flatten(),
        y=ability_sorted_incomplete.select("idx").to_numpy().flatten(),
        xerr=[
            ability_sorted_incomplete.select("ability_mean").to_numpy().flatten()
            - ability_sorted_incomplete.select("ability_hdi_lower")
            .to_numpy()
            .flatten(),
            ability_sorted_incomplete.select("ability_hdi_upper").to_numpy().flatten()
            - ability_sorted_incomplete.select("ability_mean").to_numpy().flatten(),
        ],
        fmt="o",
        color="red",
        markersize=1,
        elinewidth=0.5,
        capsize=0,
        alpha=0.5,
        label="Incomplete",
    )

    ax1.axvline(x=0, color="black", linestyle="--", alpha=0.2)
    ax1.set_xlabel(r"Ability Parameter ($\alpha$)", fontsize=LABEL_SIZE)
    ax1.set_yticks([])
    ax1.grid(True, alpha=0.1)
    ax1.set_title("(a) Participant Abilities with 94% HDI", fontsize=TITLE_SIZE)
    ax1.tick_params(axis="x", labelsize=TICK_SIZE)
    ax1.legend(fontsize=LEGEND_SIZE)

    # Difficulty Forest Plot
    difficulty_sorted = plot_df_difficulty.sort("diff_mean")
    ax2.errorbar(
        x=difficulty_sorted.select("diff_mean").to_numpy().flatten(),
        y=range(len(difficulty_sorted)),
        xerr=[
            difficulty_sorted.select("diff_mean").to_numpy().flatten()
            - difficulty_sorted.select("diff_hdi_lower").to_numpy().flatten(),
            difficulty_sorted.select("diff_hdi_upper").to_numpy().flatten()
            - difficulty_sorted.select("diff_mean").to_numpy().flatten(),
        ],
        fmt="o",
        markersize=2,
        elinewidth=0.5,
        capsize=0,
        alpha=0.3,
    )
    ax2.axvline(x=0, color="black", linestyle="--", alpha=0.2)
    ax2.set_xlabel(r"Difficulty Parameter ($\beta$)", fontsize=LABEL_SIZE)
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.1)
    ax2.set_title("(b) Task Difficulties with 94% HDI", fontsize=TITLE_SIZE)
    ax2.tick_params(axis="x", labelsize=TICK_SIZE)

    plt.tight_layout()
    plt.savefig(
        os.path.join(basepath, "figures", f"{model_name}_forest_plots.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def logit_to_prob(logit):
    """Convert logit to probability."""
    # Convert from logit scale to probability
    prob = 1 / (1 + np.exp(-(logit)))
    return prob


def get_stats(trace, model_name, df, verbose=False):
    """Calculate and output key statistics from the IRT model trace.

    Args:
        trace: ArviZ trace object containing posterior samples
        model_name: Name of the model (used for output file naming)
        verbose: Whether to print results to console
    """
    # Initialize output string to store all results
    output = []
    output.append(f"=== IRT Model Statistics: {model_name} ===\n")

    # 1. Feedback Effects (Epsilon)
    output.append("Feedback Effects")
    output.append("-" * 50)

    # Calculate epsilon statistics
    epsilon1 = trace.posterior["epsilon_one"].mean(dim=("chain", "draw")).values.item()
    epsilon2 = trace.posterior["epsilon_two"].mean(dim=("chain", "draw")).values.item()

    # Calculate probability changes for average participant/task
    change1 = logit_to_prob(epsilon1) - logit_to_prob(0)
    change2 = logit_to_prob(epsilon2) - logit_to_prob(0)

    # Calculate HDI
    epsilon1_hdi = az.hdi(trace.posterior["epsilon_one"], hdi_prob=0.94)
    epsilon2_hdi = az.hdi(trace.posterior["epsilon_two"], hdi_prob=0.94)

    # Convert to probability differences
    epsilon1_hdi_prob = (
        logit_to_prob(epsilon1_hdi.sel(hdi="lower").epsilon_one.values)
        - logit_to_prob(0),
        logit_to_prob(epsilon1_hdi.sel(hdi="higher").epsilon_one.values)
        - logit_to_prob(0),
    )
    epsilon2_hdi_prob = (
        logit_to_prob(epsilon2_hdi.sel(hdi="lower").epsilon_two.values)
        - logit_to_prob(0),
        logit_to_prob(epsilon2_hdi.sel(hdi="higher").epsilon_two.values)
        - logit_to_prob(0),
    )

    output.append(rf"First additional attempt ($\epsilon_1 ={epsilon1:.2f}$):\n")
    output.append(f"- Probability increase: {change1:.1%}")
    output.append(
        f" (94% HDI: [{epsilon1_hdi_prob[0]:.1%}, {epsilon1_hdi_prob[1]:.1%}])\n\n"
    )
    output.append(rf"Second additional attempt ($\epsilon_2 ={epsilon2:.2f}$):\n")
    output.append(f"- Probability increase: {change2:.1%}")
    output.append(
        f" (94% HDI: [{epsilon2_hdi_prob[0]:.1%}, {epsilon2_hdi_prob[1]:.1%}])\n\n"
    )

    # 2. Task Type Differences
    output.append("Task Type Differences")
    output.append("-" * 50)

    # Calculate mean difficulties
    beta_training = trace.posterior["beta"].sel(
        tasks=df.filter(pl.col("task_type") == "training")
        .select("tasks")
        .to_numpy()
        .flatten()
    )
    beta_eval = trace.posterior["beta"].sel(
        tasks=df.filter(pl.col("task_type") == "evaluation")
        .select("tasks")
        .to_numpy()
        .flatten()
    )

    beta_training_mean = beta_training.mean().values
    beta_eval_mean = beta_eval.mean().values

    # Calculate probability differences for average participant
    p_training = logit_to_prob(0 - beta_training_mean) - logit_to_prob(0)
    p_eval = logit_to_prob(0 - beta_eval_mean) - logit_to_prob(0)

    # Calculate HDI
    training_hdi = az.hdi(
        trace.posterior["beta"]
        .sel(
            tasks=df.filter(pl.col("task_type") == "training")
            .select("tasks")
            .to_numpy()
            .flatten()
        )
        .mean(dim=["tasks"])
        .values.flatten(),
        hdi_prob=0.94,
    )
    eval_hdi = az.hdi(
        trace.posterior["beta"]
        .sel(
            tasks=df.filter(pl.col("task_type") == "evaluation")
            .select("tasks")
            .to_numpy()
            .flatten()
        )
        .mean(dim=["tasks"])
        .values.flatten(),
        hdi_prob=0.94,
    )

    p_training_hdi = (
        logit_to_prob(0 - training_hdi[1]) - logit_to_prob(0),
        logit_to_prob(0 - training_hdi[0]) - logit_to_prob(0),
    )
    p_eval_hdi = (
        logit_to_prob(0 - eval_hdi[1]) - logit_to_prob(0),
        logit_to_prob(0 - eval_hdi[0]) - logit_to_prob(0),
    )

    output.append("Training Tasks:\n")
    output.append(rf"- Mean difficulty ($\beta = {beta_training_mean:.2f}$)\n")
    output.append(f"- Success probability increase: {p_training:.1%}")
    output.append(f" (94% HDI: [{p_training_hdi[0]:.1%}, {p_training_hdi[1]:.1%}])\n\n")

    output.append("Evaluation Tasks:\n")
    output.append(rf"- Mean difficulty ($\beta = {beta_eval_mean:.2f}$)\n")
    output.append(f"- Success probability increase: {p_eval:.1%}")
    output.append(f" (94% HDI: [{p_eval_hdi[0]:.1%}, {p_eval_hdi[1]:.1%}])\n\n")

    # 3. Mean Task Accuracy by Shot
    output.append("Mean Task Accuracy by Shot\n")
    output.append("-" * 50)

    # Calculate mean task accuracies
    mean_task_acc_training = trace.posterior["mean_task_acc_training"].mean(
        dim=["chain", "draw"]
    )
    mean_task_acc_training_hdi = az.hdi(
        trace.posterior["mean_task_acc_training"], hdi_prob=0.94
    )
    mean_task_acc_eval = trace.posterior["mean_task_acc_eval"].mean(
        dim=["chain", "draw"]
    )
    mean_task_acc_eval_hdi = az.hdi(
        trace.posterior["mean_task_acc_eval"], hdi_prob=0.94
    )

    # Create formatted output for each shot
    for shot_idx, shot in enumerate(mean_task_acc_training.shots.values):
        output.append(f"\n{shot}:\n")

        # Training tasks
        train_acc = mean_task_acc_training.values[shot_idx] * 100
        train_hdi_lower = (
            mean_task_acc_training_hdi.sel(hdi="lower").mean_task_acc_training.values[
                shot_idx
            ]
            * 100
        )
        train_hdi_upper = (
            mean_task_acc_training_hdi.sel(hdi="higher").mean_task_acc_training.values[
                shot_idx
            ]
            * 100
        )

        output.append(f"- Training Tasks: {train_acc:.1f}%")
        output.append(f" (94% HDI: [{train_hdi_lower:.1f}%, {train_hdi_upper:.1f}%])\n")

        # Evaluation tasks
        eval_acc = mean_task_acc_eval.values[shot_idx] * 100
        eval_hdi_lower = (
            mean_task_acc_eval_hdi.sel(hdi="lower").mean_task_acc_eval.values[shot_idx]
            * 100
        )
        eval_hdi_upper = (
            mean_task_acc_eval_hdi.sel(hdi="higher").mean_task_acc_eval.values[shot_idx]
            * 100
        )

        output.append(f"- Evaluation Tasks: {eval_acc:.1f}%")
        output.append(f" (94% HDI: [{eval_hdi_lower:.1f}%, {eval_hdi_upper:.1f}%])\n")

    # Write results to file
    output_path = os.path.join(basepath, "results", f"{model_name}_stats.md")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(output))

    # Print to console if verbose
    if verbose:
        print("\n".join(output))

    return output_path


if __name__ == "__main__":
    args = get_args()
    model_name = Path(args.model_path).stem
    model, trace = load_model(args.model_path)
    # total_params = (
    #     len(trace.posterior.participants)  # ability parameters
    #     + len(trace.posterior.tasks)  # difficulty parameters
    #     + 2  # epsilon parameters
    # )
    # print(f"Total number of parameters: {total_params}")
    plot_trace(trace, model_name)
    plot_df_ability, plot_df_difficulty, plot_df_epsilon, plot_df_epsilon_alt = (
        generate_irt_dataframes(trace)
    )
    # save IRT model parameters
    plot_df_ability.write_csv(
        os.path.join(basepath, "data", f"{model_name}_ability_parameters.csv")
    )
    plot_df_difficulty.write_csv(
        os.path.join(basepath, "data", f"{model_name}_difficulty_parameters.csv")
    )
    plot_df_epsilon_alt.write_csv(
        os.path.join(basepath, "data", f"{model_name}_epsilon_parameters.csv")
    )
    plot_irt_parameters(
        plot_df_epsilon, plot_df_ability, plot_df_difficulty, model_name
    )
    forest_plots(plot_df_ability, plot_df_difficulty, model_name)
    get_stats(trace, model_name, plot_df_difficulty, args.verbose)
