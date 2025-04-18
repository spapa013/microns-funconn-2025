import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("funconnect")


def attach_node_attrs(edge_df, node_df, node_attrs=None):
    """Attach node attributes to edge dataframe.

    Args:
        edge_df (pd.DataFrame): edge dataframe with columns `pre` and `post`.
        node_df (pd.DataFrame): node dataframe with columns `nucleus_id` and `node_attrs`.
        node_attrs (list): list of node attributes to be attached.

    Returns:
        pd.DataFrame: edge dataframe with node attributes attached.
    """
    if node_attrs:
        assert set(node_attrs).issubset(node_df.columns)
    else:
        attached_node_attrs = [
            col.replace("pre_", "") for col in edge_df.columns if col.startswith("pre_")
        ]
        node_attrs = [col for col in node_df.columns if col not in attached_node_attrs]
    edge_df = (
        edge_df.reset_index()
        .merge(
            node_df[node_attrs + ["nucleus_id"]]
            .rename(columns=lambda col: "pre_" + col)
            .rename(columns={"nucleus_id": "presyn_nucleus_id"}),
            how="left",
            validate="m:1",
        )
        .merge(
            node_df[node_attrs + ["nucleus_id"]]
            .rename(columns=lambda col: "post_" + col)
            .rename(columns={"nucleus_id": "postsyn_nucleus_id"}),
            how="left",
            validate="m:1",
        )
    )
    return edge_df


def commonize_pre(df, groupby="proj_hva", across="population", minimum_set_group=None):
    # check if df[groupby] and df[across] is categorical
    assert df[groupby].dtype.name == "category" and df[across].dtype.name == "category"
    gdfs = []
    for g, gdf in df.groupby(groupby, observed=True):
        if minimum_set_group:
            assert minimum_set_group in gdf[across].cat.categories
            minimum_pre_set = gdf.query(f"{across}=='{minimum_set_group}'")[
                "pre_nucleus_id"
            ].unique()
            minimum_pre_set = set(minimum_pre_set)
            for a in gdf[across].unique():
                assert minimum_pre_set.issubset(
                    gdf.query(f"{across}=='{a}'")["pre_nucleus_id"].unique()
                )
        else:
            pre_nucleus_ids = gdf.groupby(across, observed=True)[
                "pre_nucleus_id"
            ].apply(set)
            minimum_pre_set = set.intersection(*pre_nucleus_ids)
        filtered_gdf = gdf.query("pre_nucleus_id in @minimum_pre_set")
        if filtered_gdf.pre_nucleus_id.nunique() < gdf.pre_nucleus_id.nunique():
            logger.info(
                "%d/%d/%d (left/connected/total) pres are left for %s",
                filtered_gdf.pre_nucleus_id.nunique(),
                gdf.query("population=='connected'").pre_nucleus_id.nunique(),
                gdf.pre_nucleus_id.nunique(),
                g,
            )
        gdfs.append(filtered_gdf)
    return pd.concat(gdfs)


def summarize_edge_data(edge_data, strict=False):
    """Summarize edge data.
    Args:
        edge_data (pd.DataFrame): edge data
        strict (bool): whether to error if columns are missing
    Returns:
        dict: summary of edge data
    """
    summary = {}
    required_columns = ["pre_nucleus_id", "post_nucleus_id", "n_synapses", "dend_len"]

    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in edge_data.columns]
    if missing_columns:
        if strict:
            raise ValueError(f"Missing columns: {missing_columns}")
        else:
            logger.warning(
                f"Missing columns: {missing_columns}. Some statistics will not be available."
            )

    # Calculate summary statistics only for available columns
    if "pre_nucleus_id" in edge_data.columns:
        summary["n_pre"] = edge_data["pre_nucleus_id"].nunique()

    if all(col in edge_data.columns for col in ["post_nucleus_id", "n_synapses"]):
        summary["n_post"] = edge_data.query("n_synapses>0")["post_nucleus_id"].nunique()

    if all(
        col in edge_data.columns
        for col in ["post_nucleus_id", "n_synapses", "dend_len"]
    ):
        summary["n_adp"] = edge_data.query("dend_len > 0 and n_synapses == 0")[
            "post_nucleus_id"
        ].nunique()

    if all(col in edge_data.columns for col in ["post_nucleus_id", "dend_len"]):
        summary["n_same_region"] = edge_data.query("dend_len == 0")[
            "post_nucleus_id"
        ].nunique()

    if "n_synapses" in edge_data.columns:
        summary["n_syn"] = edge_data["n_synapses"].sum()
        summary["n_pre_post_pairs"] = len(edge_data.query("n_synapses>0"))
        summary["n_synapses"] = edge_data["n_synapses"].sum()

    if "dend_len" in edge_data.columns:
        summary["n_pre_adp_pairs"] = len(edge_data.query("dend_len > 0"))
        summary["n_pre_same_region_pairs"] = len(edge_data.query("dend_len == 0"))
        summary["dend_len"] = edge_data["dend_len"].sum()

    summary["n_pairs"] = edge_data.shape[0]

    return summary


def filter_edge_data(
    edge_data,
    var_criteria="index == index",
    quantity_criteria="index == index",
    n_pre_thld=0,
    n_synapse_thld=0,
    n_synapse_per_pre_thld=0,
    pre_contrib_thld=1,
    proj="proj_hva",
    filter_order=2,
):
    """
    Args:
        edge_data (pd.DataFrame): edge data
        var_criteria (str): criteria to filter edge data
        quantity_criteria (str): criteria to filter edge data
        n_pre_thld (int): threshold for number of presynaptic neurons
        n_synapse_thld (int): threshold for number of synapses
        pre_contrib_thld (float): threshold for contribution of presynaptic neurons
        proj (str): projection type
        filter_order (int): order of filtering, 1 for var_criteria first, 2 for quantity_criteria first. The result could be different for the two ordering.
    Returns:
        pd.DataFrame: filtered edge data

    """
    var_data = edge_data.query(var_criteria)

    """
    `proj` groups passing the following criteria are qualified for analysis:
    1. there are more than n_pre_thld presyns with positive dend_len in the proj group
    2. there are more than n_pre_thld presyns with non-zero synapses in the proj group
    3. there are more than n_synapses_thld synapses in the proj group
    4. no one presyn contributes more than pre_contri_thld of synapses or dend_len in the proj group
    """
    qualified_proj = (
        var_data.groupby(proj, observed=True)
        .filter(
            lambda x: (
                (
                    x.groupby("pre_nucleus_id", observed=True)["dend_len"]
                    .sum()
                    .gt(0)
                    .sum()
                    > n_pre_thld
                )  # 1
                & (
                    x.groupby("pre_nucleus_id", observed=True)["n_synapses"]
                    .sum()
                    .gt(0)
                    .sum()
                    > n_pre_thld
                )  # 2
                & (x["n_synapses"].sum() > n_synapse_thld)  # 3
                & (
                    (
                        x.groupby("pre_nucleus_id", observed=True)["n_synapses"].sum()
                        / x["n_synapses"].sum()
                    ).max()
                    < pre_contrib_thld
                )  # 4
                & (
                    (
                        x.groupby("pre_nucleus_id", observed=True)["dend_len"].sum()
                        / x["dend_len"].sum()
                    ).max()
                    < pre_contrib_thld
                )  # 4
            )
        )
        .loc[:, proj]
        .unique()
    )
    # if any proj is not qualified, warn the user
    if len(qualified_proj) < len(edge_data[proj].unique()):
        logging.warning(
            f"{len(qualified_proj)} out of {len(edge_data[proj].unique())} projections are qualified."
        )
    var_data = var_data.query(f"{proj} in @qualified_proj")

    # remove pre_nucleus_id with less than n_post_thld
    if n_synapse_per_pre_thld > 0:
        valid_pre = (
            var_data.groupby(["pre_nucleus_id", proj], observed=True)["n_synapses"]
            .sum()
            .pipe(lambda x: x[x >= n_synapse_per_pre_thld])
            .reset_index()
            .groupby(proj, observed=True)["pre_nucleus_id"]
            .apply(list)
            .to_dict()
        )
        valid_pre_count = {k: len(v) for k, v in valid_pre.items()}
        all_pre_count = {
            k: var_data.groupby(proj, observed=True)
            .get_group(k)["pre_nucleus_id"]
            .nunique()
            for k in valid_pre.keys()
        }
        logger.info(
            "Pre counts: %s",
            "\n".join([
                f"{k}: {valid_pre_count[k]}/{all_pre_count[k]}"
                for k in valid_pre.keys()
            ]),
        )
        var_data = (
            var_data.groupby(proj, observed=True)
            .apply(lambda df: df[df["pre_nucleus_id"].isin(valid_pre[df.name])])
            .reset_index(drop=True)
        )

    if filter_order == 1:
        var_data = commonize_pre(var_data, groupby=proj, across="population")
        var_data = var_data.query(quantity_criteria)
    elif filter_order == 2:
        var_data = var_data.query(quantity_criteria)
        var_data = commonize_pre(var_data, groupby=proj, across="population")
    else:
        raise ValueError(f"filter_order {filter_order} is not defined.")
    return var_data
