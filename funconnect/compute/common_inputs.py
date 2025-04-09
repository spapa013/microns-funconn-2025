# %%
import gc
import json
import logging
import os

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
from scipy import stats
from seaborn import axes_style
from seaborn import objects as so
from tqdm import tqdm
from pathlib import Path

from funconnect.utility import connectomics as cx
from funconnect.utility.file import prepare_rslt_dir
from funconnect.utility.plot import ColorPalletes, hinton
from funconnect.utility.R import model_diagnostics, export_to_r_global

logger = logging.getLogger("funconnect")

result_dir = prepare_rslt_dir(
    "/notebooks/microns-funconnect/projects/functional_connectomics/apr15/common_input/",
    remove_existing=True,
)


logger = logging.getLogger("funconnect")

# %%

glm_hva_dir = Path(
    "/notebooks/microns-funconnect/projects/functional_connectomics/apr15/glm_hva/"
)

# arguments:
params = json.load(open(glm_hva_dir / "params.json", "r"))
vars_thld = params["vars_thld"]
edge_data = pd.read_pickle(glm_hva_dir / "edge_data.pkl")
node_data = pd.read_pickle(glm_hva_dir / "node_data.pkl")
edge_data = edge_data.query(
    "pre_cc_max_cvt > .4 and post_cc_max_cvt > .4 and pre_cc_abs_cvt > .2 and post_cc_abs_cvt > .2"
)
vars = [
    "in_silico_sig_corr_cvt",
    "readout_similarity_cvt",
    "readout_location_distance_cvt",
    # "delta_ori_cvt_monet",
    "in_vivo_sig_corr",
    # "delta_ori_iv",
]
N_PARALLEL = 4


# model fitting
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()
base = importr("base")
glmmTMB = importr("glmmTMB")
performance = importr("performance")
emmeans = importr("emmeans")
broom = importr("broom.mixed")

# standardize vars
proc_data = edge_data.copy()
proc_data = cx.commonize_pre(proc_data)
proc_data.loc[:, vars] = proc_data.loc[:, vars] / proc_data.loc[:, vars].std()
proc_data = proc_data.loc[
    :,
    [
        "n_synapses",
        "synapse_size",
        "dend_len",
        "pre_nucleus_id",
        "post_nucleus_id",
        "proj_hva",
    ]
    + vars,
].copy()
proc_data["n_pairs"] = 1
model_fit_rslt = []
model_coef = []

## predict n_synapses without controlling for dend_len
# model fitting
for quantity, (numerator, denominator, query, family, formula_format) in tqdm(
    params["quantity_params"].items()
):
    data = proc_data.query(query)
    data["proj_hva"] = pd.Categorical(data["proj_hva"], ordered=False)
    vars_str = " + ".join(vars)
    vars_str = "(" + vars_str + ")"
    formula = ro.Formula(
        ro.Formula(formula_format.format(var=vars_str)),
    )

    export_to_r_global(data)
    model = glmmTMB.glmmTMB(
        formula,
        family=family,
        control=glmmTMB.glmmTMBControl(
            parallel=N_PARALLEL, optCtrl=ro.r("list(iter.max=1e3,eval.max=1e3)")
        ),
    )

    # model checks
    model_check = model_diagnostics(
        model, result_dir / f"{quantity}_model_diagnostics.png"
    )

    # model performance
    model_performance = ro.conversion.rpy2py(performance.performance(model))
    model_fit_rslt.append({
        **model_check,
        **model_performance.to_dict(orient="records")[0],
        "quantity_name": quantity,
    })
    assert model_check["converged"], "Model did not converge"

    # inference
    for _var in vars:
        emtrends = emmeans.emtrends(
            model, ro.Formula(" ~ proj_hva"), var=_var, infer=True, adjust="none"
        )
        trend = (
            (
                ro.conversion.rpy2py(
                    ro.r["tidy"](emtrends, **{"conf.int": True})
                ).rename(
                    columns={
                        _var + ".trend": "coef",
                        "conf.low": "coef_ci_low",
                        "conf.high": "coef_ci_high",
                        "p.value": "p",
                        "statistic": "z",
                    }
                )
            )
            .set_index("proj_hva")
            .assign(
                variable=_var,
                quantity_name=quantity,
                n_pre=data.groupby("proj_hva")["pre_nucleus_id"].nunique(),
                n_syn=data.groupby("proj_hva")["n_synapses"].sum(),
            )
            .drop(labels=["df", "std.error"], axis=1)
            .reset_index()
        )
        model_coef.append(trend)

    # predict
    pred = ro.r.predict(model, type="response")
    pred_name = "pred_" + numerator + "|" + denominator + "|" + quantity
    pred_df = data.assign(**{pred_name: pred})

    # merge back to edge_data
    edge_data = edge_data.merge(
        pred_df.loc[:, ["pre_nucleus_id", "post_nucleus_id", pred_name]],
        on=["pre_nucleus_id", "post_nucleus_id"],
        how="left",
    )
    edge_data[pred_name] = edge_data[pred_name].fillna(0)

# %%
model_coef = pd.concat(model_coef)
model_coef.to_csv(result_dir / "model_coef.csv")
model_fit_rslt = pd.DataFrame(model_fit_rslt)
model_fit_rslt.to_csv(result_dir / "model_fit_rslt.csv")
# save edge_data as pickle
edge_data = edge_data.reset_index(drop=True)
edge_data.to_pickle(result_dir / "edge_data_common_input.pkl")

# %%
# get responses, feature weights and readout location
# iv_sc_T = T.InVivoSigCorr(
#     scan_done_set_id="181ade12370b1d7ed1de664371b28eb6",
#     trace_filterset_id="f998f390a7d84eb6b43f769bbc04b78e",
#     resample_id="1459893ea8caec83a6b901a08e388d15",
#     offset_id="33dbc06858d00826c17ed7b1defa525f",
#     rate_id="c9f6269381328315194ae447cd7e6062",
#     pupil_max_nans=0.25,
#     season=1,
#     clip_type="oracle",
#     binning_frames=15,
# )

# is_sc_T = T.InSilicoSigCorr2(
#     model_set_id="6d86e8bb6d9f895d79a5e30893742773",
#     videoset_id="e3dd23445aaca70cb9d0d4eb8eea95ce",
# )
resp_dir = Path(
    "/notebooks/microns-funconnect/projects/functional_connectomics/apr15/example_resps/"
)
is_mean_resp_array = np.load(resp_dir / "is_mean_resp_array.npy")
is_unit_row_idx = pd.read_csv(resp_dir / "is_unit_row_idx.csv")
iv_mean_resp_array = np.load(resp_dir / "iv_mean_resp_array.npy")
iv_unit_row_idx = pd.read_csv(resp_dir / "iv_unit_row_idx.csv")

# compute pairwise signal correlation, readout similarity and readout location distance
# Sig Corr
from funconnect.utility.function import angular_dist
from funconnect.utility.compute import cross_corr
from scipy.spatial.distance import pdist, squareform


def compute_signal_correlation_matrix(
    unit_df,
    resp_array,
    unit_row_idx,
):
    unit_cols = ["animal_id", "scan_session", "scan_idx", "unit_id"]
    assert set(unit_cols) <= set(unit_df.columns)
    # get row_idx for each unit
    unit_df = unit_df.merge(unit_row_idx, how="left", validate="m:1")
    unit_df = unit_df.loc[:, unit_cols + ["row_idx"]]
    resp_array = resp_array[unit_df.row_idx.to_numpy()]
    # compute sig corr
    sig_corr = cross_corr(resp_array)
    return sig_corr


node_data = node_data.reset_index(drop=True)
M_sc_iv = compute_signal_correlation_matrix(
    node_data, iv_mean_resp_array, iv_unit_row_idx
)
M_sc_is = compute_signal_correlation_matrix(
    node_data, is_mean_resp_array, is_unit_row_idx
)

# Readout similarity
readout = np.stack(node_data.readout_cvt.to_numpy()).reshape(len(node_data), -1)
# compute pairwise cosine similarity
M_fw = (
    readout
    @ readout.T
    / (
        np.linalg.norm(readout, axis=1)[:, None]
        @ np.linalg.norm(readout, axis=1)[:, None].T
    )
)

# Readout location distance
pos = np.stack(node_data.position_stim_cvt.to_numpy()).reshape(len(node_data), -1)
# parameters
monitor_height = 31.0
monitor_width = 55.2
distance_from_monitor = 15.0
# compute distance for non-nan rf centers
not_nan = ~np.isnan(pos).any(axis=1)
pos = np.c_[pos, np.zeros((len(pos), 1))]
pos[:, 0] = pos[:, 0] * monitor_width / 2
pos[:, 1] = pos[:, 1] * monitor_height / 2
eye = np.zeros_like(pos)
eye[:, 2] = distance_from_monitor
dist_12 = squareform(pdist(pos[not_nan], metric="euclidean"))
dist_eye = np.linalg.norm(eye[not_nan] - pos[not_nan], axis=1)
dist_eye1 = np.tile(dist_eye, (len(dist_eye), 1))
dist_eye2 = dist_eye1.T
dist_ang = (
    np.arccos((dist_eye1**2 + dist_eye2**2 - dist_12**2) / (2 * dist_eye1 * dist_eye2))
    / np.pi
    * 180
)
# add back nan
M_rf = np.zeros((len(pos), len(pos)))
M_rf[np.ix_(not_nan, not_nan)] = dist_ang

# check if these are the same as the ones in the edge_data
check = []
edge_data_check = edge_data.sample(1000)
for edge in tqdm(edge_data_check.itertuples(), total=len(edge_data_check)):
    node_idx_1 = node_data.query("nucleus_id == @edge.pre_nucleus_id").index[0]
    node_idx_2 = node_data.query("nucleus_id == @edge.post_nucleus_id").index[0]
    if not (
        (np.isclose(edge.in_vivo_sig_corr, M_sc_iv[node_idx_1, node_idx_2], atol=1e-6))
        and (
            np.isclose(
                edge.in_silico_sig_corr_cvt, M_sc_is[node_idx_1, node_idx_2], atol=1e-6
            )
        )
        and (
            np.isclose(
                edge.readout_similarity_cvt, M_fw[node_idx_1, node_idx_2], atol=1e-6
            )
        )
        and (
            np.isclose(
                edge.readout_location_distance_cvt,
                M_rf[node_idx_1, node_idx_2],
                atol=1e-6,
            )
        )
    ):
        raise ValueError("Mismatch")


# %%
# Higher Order Connectivity Analysis:
# for each projection type, for each presyn, compute:
# 1. the mean (weighted) pairwise signal correlation among all postgsyns/adps/random neurons
# 2. the mean (weighted) pairwise signal correlation between the presyn and all postgsyns/adps/random neurons

n_post_thld = 10
rslt = []
# add corr matrix index
edge_data = edge_data.merge(
    node_data[["nucleus_id"]]
    .reset_index()
    .rename(columns={"nucleus_id": "post_nucleus_id", "index": "post_idx"}),
    validate="m:1",
    how="left",
).merge(
    node_data[["nucleus_id"]]
    .reset_index()
    .rename(columns={"nucleus_id": "pre_nucleus_id", "index": "pre_idx"}),
    validate="m:1",
    how="left",
)

M_mapping = {
    "in_vivo_sig_corr": M_sc_iv,
    "in_silico_sig_corr_cvt": M_sc_is,
    "readout_similarity_cvt": M_fw,
    "readout_location_distance_cvt": M_rf,
}

edge_data["population"] = edge_data["population"].astype(
    pd.CategoricalDtype(categories=["connected", "adp", "random"], ordered=True)
)
edge_data["n_pairs"] = 1

pop_weight = (
    ("connected", "n_synapses"),
    ("adp", "dend_len"),
    ("all", "n_pairs"),
    ("adp_ltl", "pred_n_synapses|dend_len|synapse_density"),
    ("random_ltl", "pred_dend_len|n_pairs|ld_density_control"),
    ("total_ltl", "pred_n_synapses|n_pairs|n_synapses"),
)

with tqdm(
    total=len(edge_data.groupby(["proj_hva", "pre_nucleus_id"]))
    * len(M_mapping)
    * len(pop_weight)
) as pbar:
    for (proj, pre), proj_data in edge_data.groupby([
        "proj_hva",
        "pre_nucleus_id",
    ]):
        if len(proj_data.query('population=="connected"')) < n_post_thld:
            pbar.update(len(M_mapping) * len(pop_weight))
            continue
        for pop, weight in pop_weight:
            data = proj_data.query(f"`{weight}` > 0").copy()
            weight = data[weight]
            for M_key, M in M_mapping.items():
                pbar.set_description(
                    f"proj: {proj}, pre: {pre}, pop: {pop}, var: {M_key}"
                )

                # 1. the mean (weighted) pairwise signal correlation among all postgsyns/adps/random neurons
                post_corr_idx = data.post_idx
                assert post_corr_idx.is_unique
                post_corr_idx = post_corr_idx.to_numpy()
                post_corr = M[post_corr_idx][:, post_corr_idx]
                post_corr = post_corr[np.triu_indices_from(post_corr, k=1)]
                weight_mat = np.outer(weight, weight)
                weight_post = weight_mat[np.triu_indices_from(weight_mat, k=1)]
                rslt.append({
                    "population": pop,
                    "proj_hva": proj,
                    "pre_nucleus_id": pre,
                    "pre_mean": np.average(post_corr, weights=weight_post),
                    "n_post": len(post_corr_idx),
                    "type": "post",
                    "var": M_key,
                })

                # 2. the mean (weighted) pairwise signal correlation between the presyn and all postgsyns/adps/random neurons
                pre_corr_idx = data.pre_idx.to_numpy()
                pre_post_corr = M[pre_corr_idx, post_corr_idx]
                rslt.append({
                    "population": pop,
                    "proj_hva": proj,
                    "pre_nucleus_id": pre,
                    "pre_mean": np.average(pre_post_corr, weights=weight),
                    "n_post": len(post_corr_idx),
                    "type": "pre_post",
                    "var": M_key,
                })
                pbar.update()
rslt = pd.DataFrame(rslt)
rslt.to_feather(result_dir / "pre_mean_corr.feather")

# %%
# DEV: try cumulants
# load simulated graph
