import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from src.benchmarks import sphere_gw_distance
import json
import pathlib

PLOT_PATH = pathlib.Path("./plots")

def plot_trial_outcomes(trial_outcome_path: pathlib.Path):
    """Plots the content of the json file stored at the inputted path. The json
    file should be generated by the function `benchmarks.benchmarking_run`.

    Args:
        trial_outcome_path (pathlib.Path): The path to the json file. 
    """
    trial_dict = json.load(open(trial_outcome_path, "r"))

    metadata = trial_dict["metadata"]
    sphere_dim_1 = metadata["sphere_dimension_1"]
    sphere_dim_2 = metadata["sphere_dimension_2"]
    sampling_strategy = metadata["subsampling_strategy"]
    n_trials = metadata["n_trials"]

    trial_outcomes = trial_dict["data"]
    new_trial_outcome_dict = {
        (outerKey, innerKey): values
        for outerKey, innerDict in trial_outcomes.items()
        for innerKey, values in innerDict.items()
    }
    trial_outcomes_df = (
        pd.DataFrame(new_trial_outcome_dict)
        .T.reset_index(names=["Sample Size", "Trial"])
        .rename(
            columns={"pot_estimate": "POT", "ott_estimate_reg0.01": "OTT (reg=0.01)"}
        ).astype(float)
    )

    lower_quantile = (
        trial_outcomes_df[["Sample Size", "POT", "OTT (reg=0.01)"]]
        .groupby("Sample Size")
        .agg(lambda x: np.quantile(x, 0.1))
    )

    upper_quantile = (
        trial_outcomes_df[["Sample Size", "POT", "OTT (reg=0.01)"]]
        .groupby("Sample Size")
        .agg(lambda x: np.quantile(x, 0.9))
    )

    mean = (
        trial_outcomes_df[["Sample Size", "POT", "OTT (reg=0.01)"]]
        .groupby("Sample Size")
        .mean()
    )

    fig, ax = plt.subplots()
    sns.lineplot(
        data=mean,
        x="Sample Size",
        y="POT",
        label="POT (CGD, no reg)",
        color="C0",
        ax=ax,
    )
    sns.lineplot(
        data=mean,
        x="Sample Size",
        y="OTT (reg=0.01)",
        label="OTT (reg=0.01)",
        color="C1",
        ax=ax,
    )
    plt.fill_between(
        lower_quantile.index,
        lower_quantile["POT"],
        upper_quantile["POT"],
        color="C0",
        alpha=0.3,
    )
    plt.fill_between(
        lower_quantile.index,
        lower_quantile["OTT (reg=0.01)"],
        upper_quantile["OTT (reg=0.01)"],
        color="C1",
        alpha=0.3,
    )

    ax.axhline(
        sphere_gw_distance(sphere_dim_1, sphere_dim_2),
        color="black",
        linestyle="--",
        label="True Distance",
    )

    ax.legend()

    ax.set(
        title=f"Distance Between Random Points on $S^{sphere_dim_1}$ and $S^{sphere_dim_2}$",
        xlabel="Number of Sampled Points",
        ylabel="Computed Distance",
    )

    plt.savefig(PLOT_PATH / f"{sampling_strategy}_trials" / f"n_{n_trials}_d_{sphere_dim_1}_d{sphere_dim_2}.png")