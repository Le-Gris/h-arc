# Bayesian Item Response Theory (IRT) Model

This directory contains the implementation of a Bayesian Item Response Theory (IRT) model used to analyze participant performance on the Abstraction and Reasoning Corpus (ARC) tasks. The model is implemented in Python using the `pymc` library.

## Model Overview

Intuitively, IRT models disambiguate between different latent variables that are hypothesized to drive probability of success on tasks within a test. Due to the limitations of random task-participant pairings in a finite experimental setup, certain tasks/participants can potentially be under- or overestimated with respect to ground-truth difficulty/ability when simply considering empirical success rate. Fitting an IRT model using a Bayesian framework allowed us to extract credible intervals for each parameter, which is useful for dealing with the inherent uncertainty of empirical data in a principled way. Through Bayesian data imputation, missing values were treated as additional parameters to be estimated and were sampled, conditioned on values of the model parameters during inference. Additionally, the inferred item difficulties allow us to examine difficulty distributions across ARC tasks and datasets, as well as lay out a task difficulty ordering to get a better sense of which kinds of tasks are easier or harder for people. Participant and item difficulties were given $\mathcal{N}(0, \sigma_{\alpha})$ and $\mathcal{N}(0, \sigma_{\beta})$ priors with $\sigma_{\alpha}, \sigma_{\beta} \sim \mathcal{N}^+(1)$ hyperpriors. The feedback effect was modeled as follows: $\gamma_0 = 0$, $\gamma_1 \sim \mathcal{N}^+(1)$ and $\gamma_2 = \gamma_1 + \delta$, where $\delta \sim \mathcal{N}^+(1)$.

## Mean Probability of Success

The model calculates the mean probability of success for both the training and evaluation sets for each attempt. This is computed by averaging the predicted probabilities over all participants and tasks within a given set.

The formula for the mean probability of success is:
$P_{\text{set}}(k) = \frac{1}{N_p N_t} \sum_{i=1}^{N_p} \sum_{j \in \mathcal{T}_{\text{set}}} \text{logit}^{-1}(\hat{\alpha}_i - \hat{\beta}_j + \hat{\gamma}_k)$ where $\mathcal{T}_{\text{set}}$ represents either training or evaluation tasks, $k \in \{0,1,2\}$ is the attempt number and $P_{\text{set}}(k)$ is the mean probability of success for the training or evaluation set at attempt $k$.

In this equation:

- $\hat{\alpha}_i$ is the estimated ability for participant $i$.
- $\hat{\beta}_j$ is the estimated difficulty for task $j$.
- $\hat{\gamma}_k$ is the estimated effect for attempt $k$.

## Implementation

The model is defined in `bayesian_IRT.py`. The script takes a DataFrame of participant responses and fits the IRT model using PyMC, returning the model object and the inference trace.
