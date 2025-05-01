# filepath: /src/microns-funconn-2025/funconnect/compute/common_inputs.py
"""
Common Inputs Analysis Script

This module analyzes common inputs in neural connectivity data.
Can be run as a script with command line arguments for data_dir and result_dir.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import statsmodels.api as sm
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from scipy import stats
from tqdm import tqdm

from funconnect.stats.variable import Variables
from funconnect.utility.compute import cross_corr
from funconnect.utility.connectomics import attach_node_attrs, filter_edge_data
from funconnect.utility.pandas_util import set_dtype
from funconnect.utility.R import model_diagnostics

# Constants
N_PARALLEL = 4

# Setup R environment
pandas2ri.activate()
base = importr("base")
glmmTMB = importr("glmmTMB")
performance = importr("performance")
emmeans = importr("emmeans")
broom = importr("broom.mixed")

# Setup logging
logger = logging.getLogger("funconnect")


def setup_variables():
    """Define the variables used in the analysis."""
    variables = [
        # main figure
        Variables.IN_SILICO_SIG_CORR_CVT,
        Variables.READOUT_SIMILARITY_CVT,
        Variables.READOUT_LOCATION_DISTANCE_CVT,
        Variables.IN_VIVO_SIG_CORR,
    ]
    criteria = Variables.IN_SILICO_SIG_CORR_CVT.value.criteria or "index == index"
    return variables, criteria


def load_and_prepare_data(data_dir, variables, criteria):
    """Load and prepare the data for analysis."""
    logger.info(f"Loading data from {data_dir}")

    # Load data
    edge_data = pd.read_pickle(data_dir / "edge_data_v1.pkl")
    node_data = pd.read_pickle(data_dir / "node_data_v1.pkl")

    # Validate node data
    assert (
        node_data.index.is_monotonic_increasing
        and node_data.index.is_unique
        and node_data.index[0] == 0
        and node_data.index[-1] == len(node_data) - 1
    ), "Node data index must start from 0 and increment by 1"

    node_data["index"] = node_data.index
    M_sc_is = cross_corr(np.vstack(node_data.in_silico_resp.to_numpy()))

    # Process edge data
    edge_data = attach_node_attrs(
        edge_data, node_data, ["cc_max_cvt", "cc_abs_cvt", "index"]
    )
    edge_data = set_dtype(edge_data)
    edge_data = filter_edge_data(edge_data=edge_data, var_criteria=criteria)

    # Standardize variables
    for var in variables:
        if var.value.col_name not in edge_data.columns:
            raise ValueError(f"{var.value.col_name} not in edge_data")
        edge_data.loc[:, var.value.col_name] = edge_data.loc[
            :, var.value.col_name
        ].transform(lambda x: x / x.std())

    return edge_data, node_data, M_sc_is


def fit_models(edge_data, variables, result_dir):
    """Fit the statistical models and add predictions to edge_data."""
    logger.info("Fitting statistical models")

    # Prepare data for model fitting
    var_cols = [variable.value.col_name for variable in variables]
    glmm_data = edge_data.loc[
        :,
        [
            "n_synapses",
            "synapse_size",
            "dend_len",
            "pre_nucleus_id",
            "post_nucleus_id",
            "proj_hva",
        ]
        + var_cols,
    ].copy()

    # Create formula for models
    var_str = " + ".join(var_cols)
    var_str = "(" + var_str + ")"
    formula_mapping = (
        ("pred_n_synapses", f"n_synapses ~ {var_str}*proj_hva + (1|pre_nucleus_id)"),
        ("pred_dend_len", f"dend_len ~ {var_str}*proj_hva + (1|pre_nucleus_id)"),
        (
            "pred_n_synapses|dend_len",
            f"n_synapses ~ {var_str}*proj_hva + (1|pre_nucleus_id) + offset(log(dend_len))",
        ),
    )

    # Fit models and generate predictions
    for pred_name, formula in formula_mapping:
        logger.info(f"Fitting model for {pred_name}")
        r_model = glmmTMB.glmmTMB(
            data=glmm_data,
            formula=ro.Formula(formula),
            family=ro.r["poisson"](link="log"),
            control=glmmTMB.glmmTMBControl(
                parallel=N_PARALLEL, optCtrl=ro.r("list(iter.max=1e3,eval.max=1e3)")
            ),
        )

        # Model diagnostics
        model_check = model_diagnostics(r_model, result_dir / "model_diagnostics.png")
        assert model_check["converged"], f"Model for {pred_name} did not converge"

        # Generate predictions
        pred = ro.r.predict(r_model, type="response")
        edge_data[pred_name] = pred

        logger.info(f"Predictions for {pred_name} added to edge_data")

    return edge_data


def analyze_connectivity(edge_data, M_sc_is, result_dir):
    """Perform higher order connectivity analysis."""
    logger.info("Performing higher order connectivity analysis")

    # Parameters
    n_post_thld = 10
    pre_mean_corr = []

    weights = [
        "n_synapses",
        "dend_len",
        "pred_n_synapses",
        "pred_dend_len",
        "pred_n_synapses|dend_len",
    ]

    # Process each projection type and presynaptic neuron
    with tqdm(
        total=len(edge_data.groupby(["proj_hva", "pre_nucleus_id"])) * len(weights)
    ) as pbar:
        for (proj, pre), proj_data in edge_data.groupby([
            "proj_hva",
            "pre_nucleus_id",
        ]):
            if len(proj_data.query('population=="Connected"')) < n_post_thld:
                pbar.update(len(weights))
                continue
            for weight_col in weights:
                data = proj_data.query(f"`{weight_col}` > 0").copy()
                weight = data[weight_col]
                pbar.set_description(f"proj: {proj}, pre: {pre}, weight: {weight_col}")

                # 1. Mean weighted pairwise signal correlation among all postsyns
                post_corr_idx = data.post_index
                assert post_corr_idx.is_unique
                post_corr_idx = post_corr_idx.to_numpy()
                post_corr = M_sc_is[post_corr_idx][:, post_corr_idx]
                post_corr = post_corr[np.triu_indices_from(post_corr, k=1)]
                weight_mat = np.outer(weight, weight)
                weight_post = weight_mat[np.triu_indices_from(weight_mat, k=1)]
                pre_mean_corr.append({
                    "weight": weight_col,
                    "proj_hva": proj,
                    "pre_nucleus_id": pre,
                    "pre_mean": np.average(post_corr, weights=weight_post),
                    "n_post": len(post_corr_idx),
                    "type": "post",
                })

                # 2. Mean weighted signal correlation between presyn and postsyns
                pre_corr_idx = data.pre_index.to_numpy()
                pre_post_corr = M_sc_is[pre_corr_idx, post_corr_idx]
                pre_mean_corr.append({
                    "weight": weight_col,
                    "proj_hva": proj,
                    "pre_nucleus_id": pre,
                    "pre_mean": np.average(pre_post_corr, weights=weight),
                    "n_post": len(post_corr_idx),
                    "type": "pre_post",
                })
                pbar.update()

    # Convert to DataFrame
    pre_mean_corr = pd.DataFrame(pre_mean_corr)

    return pre_mean_corr


def perform_statistical_analysis(pre_mean_corr, result_dir):
    """Perform statistical analysis on the correlation data."""
    logger.info("Performing statistical analysis")

    # Validate data integrity
    assert not pre_mean_corr.isna().any().any(), "NaN values found in correlation data"

    # For each proj_hva value, check that the set of pre_nucleus_id is identical across all weight groups
    for proj_hva in pre_mean_corr["proj_hva"].unique():
        # Get all unique sets of pre_nucleus_id for each weight group within this proj_hva
        pre_ids_by_weight = {}
        for weight, group_data in pre_mean_corr[
            pre_mean_corr["proj_hva"] == proj_hva
        ].groupby("weight"):
            pre_ids_by_weight[weight] = set(group_data["pre_nucleus_id"])

        # Check that all these sets are identical
        first_weight = list(pre_ids_by_weight.keys())[0]
        for weight, pre_ids in pre_ids_by_weight.items():
            assert pre_ids == pre_ids_by_weight[first_weight], (
                f"Pre-nucleus IDs differ between weight groups for proj_hva={proj_hva}"
            )

    # Define effect comparisons
    comparisons = [
        ("total effect", ("n_synapses", "pred_n_synapses")),
        ("axonal effect", ("dend_len", "pred_dend_len")),
        ("synaptic effect", ("n_synapses", "pred_n_synapses|dend_len")),
    ]

    # Perform statistical tests
    stats_results = []
    for effect, (weight1, weight2) in comparisons:
        for (_proj, _type), data in pre_mean_corr.groupby([
            "proj_hva",
            "type",
        ]):
            data = data.set_index("pre_nucleus_id")
            data1 = data.query("weight==@weight1")["pre_mean"]
            data2 = data.query("weight==@weight2")["pre_mean"]
            assert data1.index.equals(data2.index) and data1.index.is_unique, (
                f"Indices must match for {_proj}, {_type}"
            )
            t, p = stats.wilcoxon(data1, data2)
            stats_results.append({
                "effect": effect,
                "contrast": f"{weight1} vs {weight2}",
                "t": t,
                "p": p,
                "proj_hva": _proj,
                "type": _type,
                "n": len(data1),
            })

    # Adjust p-values for multiple comparisons
    stats_results = pd.DataFrame(stats_results)
    stats_results["p_adj"] = stats_results.groupby(["type"])["p"].transform(
        lambda x: sm.stats.multipletests(x, method="fdr_bh")[1]
    )

    return stats_results


def main(data_dir, result_dir):
    """Main function to execute the complete analysis pipeline."""
    # Create result directory if it doesn't exist
    result_dir.mkdir(parents=True, exist_ok=True)

    # Setup analysis variables
    variables, criteria = setup_variables()

    # Load and prepare data
    edge_data, node_data, M_sc_is = load_and_prepare_data(data_dir, variables, criteria)

    # Fit models and add predictions
    edge_data = fit_models(edge_data, variables, result_dir)

    # Perform connectivity analysis
    pre_mean_corr = analyze_connectivity(edge_data, M_sc_is, result_dir)

    # Perform statistical analysis
    stats_results = perform_statistical_analysis(pre_mean_corr, result_dir)

    logger.info("Analysis completed successfully")

    # Save results
    pre_mean_corr.to_feather(result_dir / "pre_mean_corr.feather")
    logger.info(
        f"Pre-mean correlation data saved to {result_dir / 'pre_mean_corr.feather'}"
    )
    stats_results.to_csv(result_dir / "stats_results.csv", index=False)
    logger.info(f"Statistical results saved to {result_dir / 'stats_results.csv'}")

    return


if __name__ == "__main__":
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description="Common Inputs Analysis for neural connectivity data."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/data"),
        help="Directory containing input data (default: /data)",
    )
    parser.add_argument(
        "--result_dir",
        type=Path,
        default=None,
        help="Directory to save results (default: <project_dir>/results/common_inputs)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Set default result_dir if not provided
    if args.result_dir is None:
        root_dir = Path(__file__).parent.parent
        args.result_dir = root_dir / "results" / "common_inputs"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run main function
    logger.info(
        f"Starting analysis with data_dir={args.data_dir}, result_dir={args.result_dir}"
    )
    main(args.data_dir, args.result_dir)
