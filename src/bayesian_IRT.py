import pandas as pd
import pymc as pm
import numpy as np


def bayes_irt(df, n_samples=5000, tune=1000, seed=0):
    """
    Bayesian Item Response Theory model
    :param df: pandas DataFrame
    :param n_samples: int, number of samples
    :param tune: int, number of burn in samples
    :param seed: int, seed for random number generator
    :return: model, trace
    """

    np.random.seed(seed)
    # create indices for participants
    participants_idx, participants = pd.factorize(df["hashed_id"], sort=True)

    # create task index
    task_idx, tasks = pd.factorize(df["task_name"], sort=True)

    # task type
    task_type_idx, _ = pd.factorize(
        df["task_type"], sort=True
    )  # sorted means evaluation=0, training=1

    # training and eval tasks
    training_task_idx = task_idx[task_type_idx == 1]
    training_task_idx = np.unique(training_task_idx)
    training_tasks = tasks[training_task_idx]
    eval_task_idx = task_idx[task_type_idx == 0]
    eval_task_idx = np.unique(eval_task_idx)
    eval_tasks = tasks[eval_task_idx]

    # training and eval participants
    training_participants_idx = participants_idx[task_type_idx == 1]
    training_participants_idx = np.unique(training_participants_idx)
    eval_participants_idx = participants_idx[task_type_idx == 0]
    eval_participants_idx = np.unique(eval_participants_idx)

    # coords
    coords = {
        "participants": participants,
        "tasks": tasks,
        "shots": ["1-shot", "2-shots", "3-shots"],
        "obs": np.arange(len(df)),
        "training_tasks": training_tasks,
        "eval_tasks": eval_tasks,
    }

    with pm.Model(coords=coords) as model:

        # hyperpriors
        mu_alpha = 0
        mu_beta = 0

        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1)

        # Ability (alpha) for each participant
        alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, dims="participants")

        # Difficulty (beta) for each task
        beta = pm.Normal("beta", mu=mu_beta, sigma=sigma_beta, dims="tasks")

        # Learning rate (epsilon) for each shot
        epsilon_zero = 0
        epsilon_one = pm.HalfNormal("epsilon_one", sigma=1)
        delta = pm.HalfNormal("epsilon_delta", sigma=1)
        epsilon_two = epsilon_one + delta
        pm.Deterministic("epsilon_two", epsilon_two)

        # Stack epsilons as a vector
        epsilon = pm.math.stack([epsilon_zero, epsilon_one, epsilon_two])

        # Likelihood
        p = pm.math.invlogit(
            alpha[participants_idx, None] - beta[task_idx, None] + epsilon[None, :]
        )
        observed = df[["1-shot", "2-shots", "3-shots"]].values
        pm.Bernoulli("outcomes", p=p, observed=observed, dims=("obs", "shots"))

        # Calculate logits for every participant on every train task
        logits_all_training = (
            alpha[:, None, None]  # Participant abilities (N_participants, 1, 1)
            - beta[
                None, training_task_idx, None
            ]  # Task difficulties (1, N__training_tasks, 1)
            + epsilon[None, None, :]  # Learning rates (1, 1, N_shots)
        )

        # Apply invlogit to get probabilities
        p_all_training = pm.math.invlogit(logits_all_training)
        pm.Deterministic(
            "mean_task_acc_training",
            p_all_training.mean(
                axis=(0, 1)
            ),  # Average across participants and tasks for each attempt
            dims="shots",
        )

        # Calculate logits for every participant on every eval task
        logits_all_eval = (
            alpha[:, None, None]  # Participant abilities (N_participants, 1, 1)
            - beta[None, eval_task_idx, None]  # Task difficulties (1, N_eval_tasks, 1)
            + epsilon[None, None, :]  # Learning rates (1, 1, N_shots)
        )

        # Apply invlogit to get probabilities
        p_all_eval = pm.math.invlogit(logits_all_eval)
        pm.Deterministic(
            "mean_task_acc_eval",
            p_all_eval.mean(
                axis=(0, 1)
            ),  # Average across participants and tasks for each attempt
            dims="shots",
        )

        # Sampling
        trace = pm.sample(
            n_samples,
            tune=tune,
            return_inferencedata=True,
            random_seed=seed,
        )

    return model, trace
