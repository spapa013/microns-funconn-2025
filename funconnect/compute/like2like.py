import logging
import os
from pathlib import Path
from typing import List
from tqdm import tqdm
import numpy as np
import pandas as pd

from funconnect.stats.like2like import (
    glmm_analysis,
    trend_analysis,
    presyn_mean_analysis,
)
from funconnect.stats.quantity import Quantities, Quantity
from funconnect.stats.variable import Variables, Variable
from funconnect.utility.performance import timer
from funconnect.utility.file import prepare_rslt_dir
import argparse

logger = logging.getLogger("funconnect")

n_parallel = int(os.getenv("N_PARALLEL", 6))  # number of parallel processes
root_dir = Path(__file__).parent.parent
data_dir = "/data"
result_dir = root_dir / "results" / "like2like"

variable_names: List[str] = [
    # main figure
    Variables.IN_SILICO_SIG_CORR_CVT.name,
    Variables.READOUT_SIMILARITY_CVT.name,
    Variables.READOUT_LOCATION_DISTANCE_CVT.name,
    # supplementary figure
    Variables.IN_VIVO_SIG_CORR.name,
    Variables.DELTA_ORI_CVT_MONET_FULL.name,
    Variables.DELTA_ORI_IV.name,
    Variables.READOUT_LOCATION_DISTANCE_STA.name,
]
quantity_names: List[str] = [
    Quantities.SYNAPSE_DENSITY.name,
    Quantities.LD_DENSITY_CONTROL.name,
]


def run_like2like_analysis(
    variables: List[Variable],
    quantities: List[Quantity],
    data_dir: Path,
    result_dir: Path,
    n_parallel: int,
) -> None:
    """
    Run the analysis pipeline for the given variables and quantities.

    Parameters:
    var_key (List[str]): List of variable keys to be analyzed.
    quantity_key (List[str]): List of quantity keys to be analyzed.
    result_dir (Path): Directory to store the results.
    n_parallel (int): Number of parallel processes to use.
    """
    # load data
    edge_data = pd.read_pickle(data_dir / "edge_data_v1.pkl")

    # compute presynaptic mean of different variables
    area_dir = Path(result_dir / "area")
    os.makedirs(area_dir, exist_ok=True)
    with timer("Computing presynaptic mean"):
        presyn_mean_analysis(
            edge_data,
            variables,
            n_synapse_per_pre_thld=10,
            result_dir=area_dir,
        )

    # fit GLMMs
    with timer("Fitting GLMMs for area-wise analysis"):
        glmm_analysis(
            edge_data=edge_data,
            proj="proj_hva",
            quantity_list=quantities,
            variable_list=variables,
            n_pre_thld=5,
            n_synapse_thld=30,
            pre_contrib_thld=0.5,
            result_dir=area_dir,
            n_parallel=n_parallel,
        )

    layer_dir = Path(result_dir / "area_layer")
    os.makedirs(layer_dir, exist_ok=True)
    with timer("Fitting GLMMs for area- and layer-wise analysis"):
        glmm_analysis(
            edge_data=edge_data,
            proj="proj_hva_layer",
            quantity_list=quantities,
            variable_list=variables,
            n_pre_thld=5,
            n_synapse_thld=30,
            pre_contrib_thld=0.5,
            result_dir=layer_dir,
            n_parallel=n_parallel,
        )

    # compute observed trend and estimate variance through bootstrapping
    dvar_bins = {
        Variables.IN_SILICO_SIG_CORR_CVT.name: np.arange(-1, 1, 0.1),
        Variables.IN_VIVO_SIG_CORR.name: np.arange(-1, 1, 0.1),
    }
    with timer("Computing bootstrapped statistics for area-wise analysis"):
        trend_analysis(
            edge_data,
            proj="proj_hva",
            quantity_list=quantities,
            variable_list=variables,
            vars_bins_dict=dvar_bins,
            n_synapse_thld=30,
            n_pre_thld=5,
            pre_contrib_thld=0.5,
            bootstrap_n=1000,
            root_seed=42,
            result_dir=area_dir,
            n_parallel=n_parallel,
            filter_order=2,
        )


if __name__ == "__main__":
    # set logging level to INFO
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run like2like analysis.")
    parser.add_argument(
        "--variable_name",
        nargs="+",
        default=variable_names,
        help="List of variable names to be analyzed.",
    )
    parser.add_argument(
        "--quantity_name",
        nargs="+",
        default=quantity_names,
        help="List of quantity names to be analyzed.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=data_dir,
        help="Directory containing the input data.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=result_dir,
        help="Directory to store the results.",
    )
    args = parser.parse_args()

    # run area-wise analysis
    data_dir = Path(args.data_dir)
    result_dir = Path(args.result_dir)
    quantities = [Quantities[q].value for q in args.quantity_name]
    variables = [Variables[v].value for v in args.variable_name]
    run_like2like_analysis(variables, quantities, data_dir, result_dir, n_parallel)

# Example usage:
# python like2like.py --data_dir /path/to/data --result_dir /path/to/results --var_key var1 var2 --quantity_key quantity1 quantity2
