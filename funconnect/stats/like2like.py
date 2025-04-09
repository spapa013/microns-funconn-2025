import itertools
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.patches import Patch
from scipy import stats
from tqdm import tqdm

from funconnect.stats.glmm import Glmm, GlmmRslt
from funconnect.stats.quantity import Quantities, Quantity
from funconnect.stats.variable import Variable, Variables
from funconnect.utility.pandas_util import read_csv
from funconnect.utility.plot import min_max_labels, rcParams
from funconnect.utility.stats import multipletests
from funconnect.utility.connectomics import summarize_edge_data, filter_edge_data

logger = logging.getLogger("funconnect")


# =============================================================================
# Mean-based Analysis
# =============================================================================
def compute_presyn_mean(
    edge_data: pd.DataFrame,
    variable_list: List[Variable],
    n_synapse_per_pre_thld: int,
):
    """
    Compute the mean of presynaptic neurons for each projection and population.
    """
    presyn_mean = []
    pbar = tqdm(total=len(variable_list) * len(edge_data["population"].unique()))
    for variable in variable_list:
        vdata = filter_edge_data(
            edge_data,
            var_criteria=variable.criteria,
            n_synapse_per_pre_thld=n_synapse_per_pre_thld,
        )
        for population, pdata in vdata.groupby("population"):
            pbar.set_description(
                f"Computing presynaptic mean for {Variables.get_name(variable)} and {population}"
            )
            presyn_mean.append(
                pdata.groupby(["proj_hva", "pre_nucleus_id"], observed=True)[
                    variable.col_name
                ]
                .agg(
                    variable_mean=lambda x: np.average(x),
                )
                .reset_index()
                .assign(
                    variable_name=Variables.get_name(variable),
                    population=population,
                )
            )
            pbar.update(1)
    presyn_mean = pd.concat(presyn_mean).reset_index(drop=True)
    return presyn_mean


def compare_presyn_mean(presyn_mean):
    """
    Compare the presynaptic mean of different populations.
    """
    presyn_stats = []
    population_list = presyn_mean["population"].unique()
    population_pair_list = list(itertools.combinations(population_list, 2))
    for (var, proj), data in presyn_mean.groupby(
        ["variable_name", "proj_hva"], observed=True
    ):
        for pop1, pop2 in population_pair_list:
            data1 = data.query("population==@pop1")
            data2 = data.query("population==@pop2")
            t, p = stats.ttest_rel(
                data1["variable_mean"].to_numpy(), data2["variable_mean"].to_numpy()
            )
            n = len(data1)
            mean_diff = (
                data1["variable_mean"].to_numpy() - data2["variable_mean"].to_numpy()
            )
            presyn_stats.append({
                "variable_name": var,
                "comparison": f"{pop1} vs {pop2}",
                "population_1": pop1,
                "population_2": pop2,
                "mean_diff": mean_diff.mean(),
                "mean_diff_low": mean_diff.mean()
                - mean_diff.std() / np.sqrt(len(mean_diff)),
                "mean_diff_high": mean_diff.mean()
                + mean_diff.std() / np.sqrt(len(mean_diff)),
                "p": p,
                "t": t,
                "proj_hva": proj,
                "n": n,
            })
    presyn_stats = pd.DataFrame(presyn_stats)
    presyn_stats["p_adj"] = presyn_stats.groupby("variable_name")["p"].transform(
        lambda x: multipletests(x, method="fdr_bh")
    )
    return presyn_stats


def presyn_mean_analysis(
    edge_data: pd.DataFrame,
    variable_list: List[Variable],
    n_synapse_per_pre_thld: int,
    result_dir: Path,
):
    """
    Compute the presynaptic mean and compare the mean of different populations.
    """
    presyn_mean = compute_presyn_mean(
        edge_data, variable_list, n_synapse_per_pre_thld=n_synapse_per_pre_thld
    )
    presyn_stats = compare_presyn_mean(presyn_mean)
    presyn_mean.to_feather(result_dir / "presyn_mean.feather")
    presyn_stats.to_feather(result_dir / "presyn_stats.feather")


# =============================================================================
# GLMM Analysis
# =============================================================================


def glmm_analysis_single(
    edge_data,
    proj,
    quantity: Quantity,
    variable: Variable,
    n_pre_thld,
    n_synapse_thld,
    pre_contrib_thld,
    result_dir,
    n_parallel=1,
    filter_order=1,
) -> GlmmRslt:
    var_data = filter_edge_data(
        edge_data,
        var_criteria=variable.criteria,
        quantity_criteria=quantity.query,
        n_pre_thld=n_pre_thld,
        n_synapse_thld=n_synapse_thld,
        pre_contrib_thld=pre_contrib_thld,
        proj=proj,
        filter_order=filter_order,
    )
    glmm = Glmm(
        quantity=quantity,
        data=var_data,
        variable=variable,
        proj=proj,
    )
    glmm_rslt = glmm.fit(result_dir, n_parallel=n_parallel, save=True)
    return glmm_rslt


def glmm_analysis(
    edge_data,
    proj,
    quantity_list,
    variable_list,
    n_pre_thld,
    n_synapse_thld,
    pre_contrib_thld,
    result_dir,
    n_parallel,
):
    """
    Run GLMM analysis for multiple quantities and variables.

    Parameters:
    -----------
    edge_data : pd.DataFrame
        Edge data containing all necessary columns
    proj : str
        Projection type
    quantity_list : list
        List of quantities to analyze
    variable_list : list
        List of variables to analyze
    n_pre_thld : int
        Threshold for number of presynaptic neurons
    n_synapse_thld : int
        Threshold for number of synapses
    pre_contrib_thld : float
        Threshold for contribution of presynaptic neurons
    result_dir : Path
        Directory to store results
    n_parallel : int
        Number of parallel processes to use
    """
    for quantity, variable in tqdm(
        [(q, v) for q in quantity_list for v in variable_list],
        desc="Fitting GLMMs",
    ):
        logger.info(
            f"Fitting GLMMs for {Quantities.get_name(quantity)} and {Variables.get_name(variable)}"
        )
        glmm_analysis_single(
            edge_data=edge_data,
            proj=proj,
            quantity=quantity,
            variable=variable,
            n_pre_thld=n_pre_thld,
            n_synapse_thld=n_synapse_thld,
            pre_contrib_thld=pre_contrib_thld,
            result_dir=result_dir
            / f"{Variables.get_name(variable).lower()}-{Quantities.get_name(quantity).lower()}",
            n_parallel=n_parallel,
        )
    return


def load_glmm_results(result_dir):
    """
    Load GLMM results from the result directory.
    Each result is saved in a subdirectory named {var}-{quantity.name} containing multiple CSV files.

    Parameters:
    -----------
    result_dir : Path
        Directory containing GLMM result subdirectories (e.g., 'area' or 'area_layer')

    Returns:
    --------
    dict
        Dictionary mapping (var, quantity_name) to GlmmRslt objects
    """
    glmm_rslts = {}
    for subdir in result_dir.iterdir():
        if subdir.is_dir():
            # Each subdir is named {var}-{quantity.name}
            variable_name, quantity_name = subdir.name.split("-", 1)
            variable_name, quantity_name = variable_name.upper(), quantity_name.upper()

            # Check if all required CSV files exist
            required_files = [
                "glmm_model_performance.csv",
                "glmm_model_coef.csv",
                "glmm_emtrends.csv",
                "glmm_pairwise.csv",
                "glmm_meta_info.csv",
            ]
            if all((subdir / f).exists() for f in required_files):
                # Load all CSV files
                model_performance = read_csv(
                    subdir / "glmm_model_performance.csv"
                ).iloc[0]
                model_coef = read_csv(subdir / "glmm_model_coef.csv")
                emtrends = read_csv(subdir / "glmm_emtrends.csv")
                pairwise = read_csv(subdir / "glmm_pairwise.csv")
                meta_info = read_csv(subdir / "glmm_meta_info.csv")

                # Create GlmmRslt object
                glmm_rslts[(variable_name, quantity_name)] = GlmmRslt(
                    model_performance=model_performance,
                    model_coef=model_coef,
                    emtrends=emtrends,
                    pairwise=pairwise,
                    meta_info=meta_info,
                )
    return glmm_rslts


# =============================================================================
# Bootstrap Analysis
# =============================================================================


def bootstrap_parallel(data, bs_idx_ls, worker_id, root_seed):
    rslt = []
    rng = np.random.default_rng([root_seed, worker_id])
    for bs_idx in bs_idx_ls:
        binned_data = (
            data.sample(frac=1, replace=True, random_state=rng)
            .groupby("bins")
            .agg(numerator=("numerator", "sum"), denominator=("denominator", "sum"))
            .reset_index()
            .assign(quantity=lambda x: x["numerator"] / x["denominator"])
        )
        rslt.append(binned_data.loc[:, ["bins", "quantity"]])
    return rslt


def compute_binned_quantity(
    var_data,
    quantity,
    variable_col_name,
    proj,
    bins,
    n_parallel,
    bootstrap_n,
    root_seed,
):
    """ """

    # rename columns for convenience in downstream processing
    var_data["numerator"] = var_data[quantity.numerator]
    if (
        quantity.denominator is None
    ):  # if no denominator, use 1, i.e., take mean of numerator
        var_data["denominator"] = 1
    else:
        var_data["denominator"] = var_data[quantity.denominator]

    # Remove per-pre biases
    groupby_cols = [proj, "pre_nucleus_id"]
    # Remove per-pre functional similarity baseline
    var_data[variable_col_name] = var_data[variable_col_name] - var_data.groupby(
        groupby_cols
    )[variable_col_name].transform("mean")
    # Remove per-pre anatomical quantity baseline
    pre_avg = var_data.groupby(groupby_cols)["numerator"].transform(
        "sum"
    ) / var_data.groupby(groupby_cols)["denominator"].transform("sum")
    var_data["numerator"] -= pre_avg * var_data["denominator"]

    var_data = var_data.assign(
        bins=pd.cut(var_data[variable_col_name], bins=bins),
    )

    # bootstrap with parallelization
    bs_ids = np.array_split(np.arange(bootstrap_n), n_parallel)
    # group by project and bootstrap
    bs_stats = []
    for proj_name, proj_data in var_data.groupby(proj):
        binned_data = []
        with mp.Pool(n_parallel) as pool:
            for r in pool.starmap(
                bootstrap_parallel,
                [(proj_data, bs_ids[i], i, root_seed) for i in range(n_parallel)],
            ):
                binned_data.extend(r)
        binned_data = pd.concat(binned_data, ignore_index=True)
        # compute summary statistics
        binned_data = (
            binned_data.groupby("bins")
            .agg(
                mean=("quantity", "mean"),
                std=("quantity", "std"),
                ci_low=("quantity", lambda x: np.percentile(x, 2.5)),
                ci_high=("quantity", lambda x: np.percentile(x, 97.5)),
            )
            .reset_index()
        )
        meta_info = (
            proj_data.groupby("bins")
            .apply(lambda df: pd.Series(summarize_edge_data(df)))
            .reset_index()
        )
        binned_data = binned_data.merge(
            meta_info,
            on="bins",
            how="left",
        )
        binned_data[proj] = proj_name
        bs_stats.append(binned_data)
    bs_stats = pd.concat(bs_stats, ignore_index=True)
    return bs_stats


def trend_analysis(
    edge_data,
    proj,
    quantity_list,
    variable_list,
    vars_bins_dict,
    n_synapse_thld,
    n_pre_thld,
    pre_contrib_thld,
    n_parallel,
    bootstrap_n,
    root_seed,
    result_dir,
    filter_order=2,  # filter order 1 and 2 generates very slightly different results, the difference is not perceivable by eye
):
    """
    Compute observed trend and estimate variance through bootstrapping.

    Parameters:
    -----------
    edge_data : pd.DataFrame
        Edge data containing all necessary columns
    proj : str
        Projection type
    quantity_list : list
        List of quantities to analyze
    vars_criteria_dict : dict
        Dictionary mapping variable names to their filtering criteria
    vars_bins_dict : dict
        Dictionary mapping variable names to their bin edges
    n_synapse_thld : int
        Threshold for number of synapses
    n_pre_thld : int
        Threshold for number of presynaptic neurons
    pre_contrib_thld : float
        Threshold for contribution of presynaptic neurons
    n_parallel : int
        Number of parallel processes to use
    bootstrap_n : int
        Number of bootstrap iterations
    root_seed : int
        Root seed for random number generation
    result_dir : Path
        Directory to store results
    filter_order : int
        Order of filtering (1 or 2)
    """
    bs_stats = []
    for quantity, variable in tqdm(
        [(q, v) for q in quantity_list for v in variable_list],
    ):
        quantity_name = Quantities.get_name(quantity)
        variable_name = Variables.get_name(variable)
        logger.info(
            f"Computing bootstrapped statistics for {quantity_name} and {variable_name}"
        )
        if variable_name in vars_bins_dict:
            bins = vars_bins_dict[variable_name]
        else:
            # for unspecified dbins, use std of the variable to determine bin size
            scale = edge_data.query(variable.criteria)[variable.col_name].std()
            bins = [0 + i * scale for i in range(-4, 5)]  # 0 +- 4 std

        var_data = filter_edge_data(
            edge_data,
            var_criteria=variable.criteria,
            quantity_criteria=quantity.query,
            n_pre_thld=n_pre_thld,
            n_synapse_thld=n_synapse_thld,
            pre_contrib_thld=pre_contrib_thld,
            proj=proj,
            filter_order=filter_order,
        )

        bs_stats.append(
            compute_binned_quantity(
                var_data,
                quantity,
                variable.col_name,
                proj,
                bins,
                n_parallel,
                bootstrap_n,
                root_seed,
            ).assign(
                quantity_name=quantity_name,
                variable_name=variable_name,
            )
        )
    bs_stats = pd.concat(bs_stats, ignore_index=True).dropna()
    # save bootstrapped statistics as csv
    bs_stats.to_csv(result_dir / "bs_stats.csv", index=False)
    return


def load_bs_stats(result_dir):
    """
    Load bootstrapped statistics from the result directory.
    """
    return read_csv(result_dir / "bs_stats.csv")
