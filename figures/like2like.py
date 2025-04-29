# %% [python]
# import libraries
from pathlib import Path
import pandas as pd
from funconnect.stats.like2like import load_glmm_results, load_bs_stats
from funconnect.visualization import like2like
from funconnect.utility.plot import rcParams, ColorPalletes
from funconnect.utility.connectomics import attach_node_attrs
from matplotlib import pyplot as plt

plt.rcParams.update(rcParams.arial_desat)


root_dir = Path(__file__).parent.parent
# data_dir = root_dir / "data"
data_dir = Path(
    "/mnt/lab/users/zhuokun/microns-funconnect/projects/functional_connectomics/code_release/data/functional_connectomics"
)
result_dir = root_dir / "results" / "like2like"

# load data
try:
    edge_data = pd.read_pickle(data_dir / "edge_data_v1.pkl")
    node_data = pd.read_pickle(data_dir / "node_data_v1.pkl")
    edge_data = attach_node_attrs(
        edge_data,
        node_data,
        [
            "cc_max_cvt",
            "cc_abs_cvt",
            "gosi_cvt_monet_full",
            "gosi_iv",
        ],
    )
except FileNotFoundError:
    raise FileNotFoundError("File not found: edge_data_v1.pkl, node_data_v1.pkl")

# load results
try:
    presyn_mean = pd.read_feather(result_dir / "area" / "presyn_mean.feather")
    presyn_stats = pd.read_feather(result_dir / "area" / "presyn_stats.feather")
    glmm_rslts = load_glmm_results(result_dir / "area")
    bs_stats = load_bs_stats(result_dir / "area")
    glmm_rslts_layer = load_glmm_results(result_dir / "area_layer")
except FileNotFoundError:
    raise FileNotFoundError(
        "Analysis results not found, please run the analysis first with `run_analysis.py`"
    )


# %% [markdown]
# # Figure 2d

# %% [python]
like2like.plot_presyn_mean(
    presyn_mean,
    presyn_stats,
    like2like.Variables.IN_SILICO_SIG_CORR_CVT,
)

# %% [markdown]
# # Figure 2e, f
# %% [python]
like2like.plot_like2like_area(
    glmm_rslts,
    bs_stats,
    variables=[
        like2like.Variables.IN_SILICO_SIG_CORR_CVT,
    ],
    quantities=[
        like2like.Quantities.LD_DENSITY_CONTROL,
        like2like.Quantities.SYNAPSE_DENSITY,
    ],
    filtering_quantity_name=like2like.Quantities.SYNAPSE_DENSITY.name,
    palette=["k"],
    x_lim_stds=5,
)

# %% [markdown]
# # Figure 2g
# %% [python]
like2like.plot_synapse_quantity(
    variables=[like2like.Variables.IN_SILICO_SIG_CORR_CVT],
    quantity=like2like.Quantities.MEAN_SYNAPSE_SIZE,
    y_pos_lim=0.5,
    y_neg_lim=0.5,
    edge_data=edge_data,
    first_axis_side="left",
    palette=["k"],
)

# %% [markdown]
# # Figure 2h
# %% [python]
like2like.plot_synapse_quantity(
    variables=[like2like.Variables.IN_SILICO_SIG_CORR_CVT],
    quantity=like2like.Quantities.N_SYNAPSES_POSITIVE,
    y_pos_lim=0.6,
    y_neg_lim=0.3,
    edge_data=edge_data,
    first_axis_side="left",
    log=False,
    palette=["k"],
)

# %% [markdown]
# # Figure 3a, b

# %% [python]
like2like.plot_like2like_area(
    glmm_rslts,
    bs_stats,
    variables=[
        like2like.Variables.READOUT_SIMILARITY_CVT,
        like2like.Variables.READOUT_LOCATION_DISTANCE_CVT,
    ],
    quantities=[
        like2like.Quantities.LD_DENSITY_CONTROL,
        like2like.Quantities.SYNAPSE_DENSITY,
    ],
    filtering_quantity_name=like2like.Quantities.SYNAPSE_DENSITY.name,
    palette=ColorPalletes.C3,
    x_lim_stds=5,
)

# %% [markdown]
# # Figure 3c

# %% [python]
like2like.plot_synapse_quantity(
    variables=[
        like2like.Variables.READOUT_SIMILARITY_CVT,
        like2like.Variables.READOUT_LOCATION_DISTANCE_CVT,
    ],
    quantity=like2like.Quantities.N_SYNAPSES_POSITIVE,
    y_pos_lim=0.6,
    y_neg_lim=0.3,
    edge_data=edge_data,
    palette=ColorPalletes.C3,
    first_axis_side="left",
    log=False,
    figsize=[2, 2],
)

# %% [markdown]
# # Figure 3d

# %% [python]
like2like.plot_synapse_quantity(
    variables=[
        like2like.Variables.READOUT_SIMILARITY_CVT,
        like2like.Variables.READOUT_LOCATION_DISTANCE_CVT,
    ],
    quantity=like2like.Quantities.MEAN_SYNAPSE_SIZE,
    y_pos_lim=0.5,
    y_neg_lim=0.5,
    edge_data=edge_data,
    palette=ColorPalletes.C3,
    first_axis_side="left",
    figsize=[2, 2],
)

# %% [markdown]
# Figure 4a, c, e

# %% [python]
for variable in [
    like2like.Variables.IN_SILICO_SIG_CORR_CVT,
    like2like.Variables.READOUT_SIMILARITY_CVT,
    like2like.Variables.READOUT_LOCATION_DISTANCE_CVT,
]:
    like2like.plot_coef_matrix(
        glmm_rslts_layer,
        variable,
        like2like.Quantities.SYNAPSE_DENSITY,
        edge_data,
        legend_size_step=0.5,
        max_size=2.5,
        min_size=0,
    )
    plt.title(variable.name + "|" + like2like.Quantities.SYNAPSE_DENSITY.name)

# %%
# Figure 4b, d, f
# %% [python]
for variable in [
    like2like.Variables.IN_SILICO_SIG_CORR_CVT,
    like2like.Variables.READOUT_SIMILARITY_CVT,
    like2like.Variables.READOUT_LOCATION_DISTANCE_CVT,
]:
    like2like.plot_coef_matrix(
        glmm_rslts_layer,
        variable,
        like2like.Quantities.LD_DENSITY_CONTROL,
        edge_data,
        legend_size_step=5e-3,
        max_size=25e-3,
        min_size=0,
    )
    plt.title(variable.name + "|" + like2like.Quantities.LD_DENSITY_CONTROL.name)
