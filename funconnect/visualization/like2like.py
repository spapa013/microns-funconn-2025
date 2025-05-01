import logging
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from matplotlib import ticker as plticker
from matplotlib.patches import Patch

from funconnect.stats.like2like import filter_edge_data
from funconnect.stats.quantity import Quantities
from funconnect.stats.variable import Variables
from funconnect.utility.pandas_util import set_dtype
from funconnect.utility.plot import (
    ColorPalletes,
    min_max_labels,
    p_to_star,
    proj_hva_mapping,
    rcParams,
    hinton,
)


def plot_presyn_mean(presyn_mean, presyn_stats, variable: Variables):
    """
    Plot the presynaptic mean of different populations.
    """
    plt.rcParams.update(rcParams.arial_desat)
    presyn_stats["star"] = presyn_stats["p_adj"].apply(p_to_star)
    data2plot = presyn_mean.query("variable_name==@variable.name").copy()
    summary = (
        data2plot.groupby(["proj_hva", "population"], observed=True)
        .apply(
            lambda df: pd.Series({
                "mean": df["variable_mean"].mean(),
                "se_low": df["variable_mean"].mean() - df["variable_mean"].sem(),
                "se_high": df["variable_mean"].mean() + df["variable_mean"].sem(),
            })
        )
        .reset_index()
    )

    summary = set_dtype(summary).sort_values([
        "proj_hva",
        "population",
    ])

    bar_width = 0.2
    bar_spacing = 0.05
    proj_spacing = 0.4

    fig, ax = plt.subplots(1, 1, figsize=(3, 1.5), layout="constrained")
    proj_idx = np.arange(summary.proj_hva.nunique())
    proj_width = (bar_width + bar_spacing) * summary.population.nunique() + proj_spacing

    bar_positions = []
    for j, p in enumerate(summary["population"].unique()):
        for k, proj in enumerate(summary["proj_hva"].unique()):
            bar_data = summary.query(f"population=='{p}' and proj_hva=='{proj}'")
            bar_pos = k * proj_width + j * (bar_width + bar_spacing)
            ax.errorbar(
                bar_pos,
                bar_data["mean"],
                yerr=[
                    bar_data["mean"] - bar_data["se_low"],
                    bar_data["se_high"] - bar_data["mean"],
                ],
                fmt=".",
                capsize=2,
                color=f"C{j}",
                elinewidth=0.8,
                markersize=3,
            )
            bar_positions.append(
                dict(
                    proj=proj,
                    pop=p,
                    x=bar_pos,
                    y=bar_data["mean"].iloc[0],
                    y_top=bar_data["se_high"].iloc[0],
                    y_bottom=bar_data["se_low"].iloc[0],
                )
            )
    bar_positions = pd.DataFrame(bar_positions)

    y_offset_unit_data = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.07
    # add stars
    for s in presyn_stats.query("variable_name==@variable.name").itertuples():
        # find the bar position
        pos_1 = bar_positions.query(
            f"proj=='{s.proj_hva}' and pop=='{s.population_1}'"
        ).x.iloc[0]
        pos_2 = bar_positions.query(
            f"proj=='{s.proj_hva}' and pop=='{s.population_2}'"
        ).x.iloc[0]
        y_top = bar_positions.query(f"proj=='{s.proj_hva}'").y_top.max()
        if s.comparison == "Connected vs ADP":
            y_offset = 1.5 * y_offset_unit_data
        elif s.comparison == "Connected vs Same region":
            y_offset = 2.5 * y_offset_unit_data
        elif s.comparison == "ADP vs Same region":
            y_offset = 0.5 * y_offset_unit_data
        # add stars
        if s.star == "n.s.":
            ax.text(
                (pos_1 + pos_2) / 2,
                y_top + y_offset,
                s.star,
                ha="center",
                va="bottom",
                fontsize=6,
                color="black",
            )
        else:
            ax.text(
                (pos_1 + pos_2) / 2,
                y_top + y_offset - 0.8 * y_offset_unit_data,
                s.star,
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
            )
        # draw lines
        ax.plot([pos_1, pos_2], [y_top + y_offset] * 2, color="black", lw=0.5)

    ax.set_xticks(proj_idx * proj_width + bar_width + bar_spacing)
    ax.set_xticklabels([proj_hva_mapping[p] for p in summary.proj_hva.unique()])
    ax.set_ylabel(variable.value.name_latex)

    # custom legend
    plt.figlegend(
        handles=[
            Patch(color=f"C{j}", label=p)
            for j, p in enumerate(summary["population"].unique())
        ],
        loc="outside right upper",
    )
    plt.xlim(
        bar_positions.x.min() - proj_spacing,
        bar_positions.x.max() + proj_spacing,
    )
    min_max_labels(ax, False, True)


def plot_like2like_area(
    glmm_rslts,
    bs_stats,
    n_synapse_thld=10,  # minimum number of synapses per var bin
    n_pre_thld=10,  # minimum number of presynaptic neurons per var bin
    x_lim_stds=3.6,
    variables: List[Variables] = [
        Variables.READOUT_SIMILARITY_CVT,
        Variables.READOUT_LOCATION_DISTANCE_CVT,
    ],
    quantities: List[Quantities] = [
        Quantities.LD_DENSITY_CONTROL,
        Quantities.SYNAPSE_DENSITY,
    ],
    filtering_quantity_name=Quantities.SYNAPSE_DENSITY.name,
    palette=ColorPalletes.C3,
):
    variable_names = [v.name for v in variables]
    quantity_names = [q.name for q in quantities]

    # check if the variables exist in the bs_stats
    for var in variables:
        assert var.name in bs_stats["variable_name"].unique(), (
            f"Variable {var.name} not found in bs_stats"
        )
    for q in quantities:
        assert q.name in bs_stats["quantity_name"].unique(), (
            f"Quantity {q.name} not found in bs_stats"
        )
    # check if the filtering_quantity_name exists in the bs_stats
    assert filtering_quantity_name in bs_stats["quantity_name"].unique(), (
        f"Filtering quantity {filtering_quantity_name} not found in bs_stats"
    )
    # if palette is not the same length as the quantities, repeat the palette
    if len(palette) < len(quantities):
        palette = (
            palette * (len(quantities) // len(palette))
            + palette[: len(quantities) % len(palette)]
        )

    _bs_stats = bs_stats.query(
        f"variable_name in {variable_names} and quantity_name in {quantity_names}"
    )

    fig, axes = plt.subplots(
        len(quantities),
        _bs_stats["proj_hva"].nunique(),
        figsize=(
            1.5 * _bs_stats["proj_hva"].nunique(),
            1.2 * len(quantities) + 0.5 * len(variables),
        ),
    )  # quantities x proj

    # Plot the trend
    for qaxes, (qname, q_bs) in zip(
        axes, _bs_stats.groupby("quantity_name", observed=True)
    ):
        for j, (var, var_bs) in enumerate(q_bs.groupby("variable_name", observed=True)):
            for i, (ax, (proj, bs)) in enumerate(
                zip(qaxes, var_bs.groupby("proj_hva", observed=True))
            ):
                if j > 0:
                    ax = ax.twiny()

                filtered_bins = bs_stats.query(
                    (
                        "variable_name == @var and quantity_name == @filtering_quantity_name and proj_hva == @proj"
                        " and n_synapses > @n_synapse_thld and n_pre > @n_pre_thld"
                    )
                )["bins"].unique()
                # filter the data
                bs = bs.query("bins in @filtered_bins").copy()
                bs["bin_center"] = bs["bins"].apply(lambda x: x.mid)
                p = ax.plot(
                    bs["bin_center"],
                    bs["mean"],
                    c=palette[j],
                    zorder=0,
                )
                ax.fill_between(
                    bs["bin_center"],
                    bs["mean"] - bs["std"],
                    bs["mean"] + bs["std"],
                    alpha=0.3,
                    color=palette[j],
                    lw=0,
                )
                ax.scatter(
                    bs["bin_center"],
                    bs["mean"],
                    c=palette[j],
                    zorder=3,
                )

                # aes
                if j == 0:
                    ax.set_title(proj_hva_mapping[proj])
                if j % 2 == 1:
                    side, other_side = "top", "bottom"
                    position = 1 + 0.2 * (j // 2)
                else:
                    side, other_side = "bottom", "top"
                    position = 0 - 0.2 * (j // 2)

                ax.xaxis.set_ticks_position(side)
                ax.xaxis.set_label_position(side)
                ax.spines[side].set_position(("axes", position))
                ax.spines[side].set_visible(True)
                ax.spines[other_side].set_visible(False)
                ax.spines[side].set_color(p[0].get_color())
                ax.tick_params(
                    axis="x", colors=p[0].get_color(), which="both", direction="out"
                )
                ax.set_xlabel(r"$\Delta$" + Variables[var].value.name_latex)
                ax.xaxis.label.set_color(p[0].get_color())

                bin_width = bs["bins"].apply(lambda x: x.right - x.left).mean()
                bin_origin = min([min([abs(b.left), abs(b.right)]) for b in bs["bins"]])
                ax.set_xlim(
                    -x_lim_stds * bin_width + bin_origin,
                    x_lim_stds * bin_width + bin_origin,
                )

                # Get p value from glmm_rslts
                p = float(
                    glmm_rslts[(var, qname)].emtrends.query("proj_hva == @proj")[
                        "p_adj"
                    ]
                )
                ax.text(
                    0.05,
                    0.95 - 0.05 * j,
                    p_to_star(p),
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=8,
                    color=palette[j],
                )

                ax.xaxis.set_major_locator(plticker.MaxNLocator(4, min_n_ticks=3))
                ax.xaxis.set_major_formatter(plticker.StrMethodFormatter("{x:n}"))
                plt.minorticks_off()
                if Variables[var].value.inverse:
                    ax.invert_xaxis()

        plt.sca(qaxes[0])
        plt.ylabel(r"$\Delta$" + Quantities[qname].value.name_latex)

    plt.tight_layout()
    plt.show()


def flipside(side: str) -> str:
    return "right" if side == "left" else "left"


def plot_synapse_quantity(
    variables: List[Variables],
    quantity: Quantities,
    y_pos_lim,
    y_neg_lim,
    edge_data,
    log=True,
    correct_by_adp=True,
    palette=ColorPalletes.C3,
    spacing=0,
    x_padding=0.1,
    figsize=[1.8, 2],
    alpha=0.8,
    first_axis_side="right",
    mean_synapse_size=False,
):
    # all variables should have the same variable.value.criteria
    assert all(variables[0].value.criteria == v.value.criteria for v in variables), (
        "All variables should have the same variable.value.criteria"
    )
    edge_data = filter_edge_data(
        edge_data,
        var_criteria=variables[0].value.criteria,
        quantity_criteria=quantity.value.query,
        filter_order=2,
    )
    fit_summary = []
    lines = []
    labels = []
    fig, ax0 = plt.subplots(
        figsize=figsize,
        layout="tight",
    )

    var_axes = [ax0]
    for _ in enumerate(variables[1:]):
        ax = ax0.twinx()
        var_axes.append(ax)

    for i, (var, ax) in enumerate(zip(variables, var_axes)):
        offset = spacing * (i - len(variables) // 2)

        var_std = edge_data[var.value.col_name].std()
        x = (
            edge_data[quantity.value.numerator] / edge_data[quantity.value.denominator]
            if quantity.value.denominator
            else edge_data[quantity.value.numerator]
        )
        y = edge_data[var.value.col_name]
        if log:
            x = np.log10(x)
        if correct_by_adp:
            # get the residue of var ~ denominator
            data = pd.DataFrame({"adp": edge_data["dend_len"].to_numpy(), "y": y})
            y = sm.OLS.from_formula("y ~ adp", data=data).fit().resid
        data = pd.DataFrame({"x": x, "y": y})
        model_fit = sm.OLS.from_formula("y ~ x", data=data).fit()
        # get coefficient and p value
        fit_summary.append({
            "n": len(data),
            "var": var,
            "slope": model_fit.params["x"],
            "intercept": model_fit.params["Intercept"],
            "p": model_fit.pvalues["x"],
            "slope_zscore": model_fit.params["x"] / var_std,
            "slope_zscore_abs": np.abs(model_fit.params["x"] / var_std),
        })

        # visualization
        # bin data for visualization
        if quantity.value.family == "poisson":  # discrete data
            max_x = data.x.max() // 1 + 1
            min_x = data.x.min() // 1
            x_bin = np.arange(min_x - 0.5, max_x + 0.5, 1)
            data["x_bin"] = pd.cut(data["x"], bins=x_bin)
        elif quantity.value.family == "tweedie":  # continuous data
            # bin data
            data["x_bin"] = pd.cut(data["x"], bins=5)
        data["x_center"] = data.x_bin.apply(lambda x: x.mid).astype(float)
        summary = (
            data.groupby("x_center", observed=True)["y"]
            .agg(["mean", "sem", "count"])
            .reset_index()
        )
        summary = summary.query("count>10")
        scatter_plot = ax.scatter(
            summary.x_center + offset,
            summary["mean"],
            edgecolor="none",
            facecolor=palette[i],
            s=10,
            label=var.value.name_latex
            if not log
            else r"$log_{10}$" + var.value.name_latex,
            alpha=alpha,
        )
        lines.append(scatter_plot)
        labels.append(scatter_plot.get_label())
        ax.errorbar(
            summary.x_center + offset,
            summary["mean"],
            yerr=summary["sem"],
            fmt="none",
            ecolor=palette[i],
            capsize=3,
            elinewidth=1,
            markeredgewidth=1,
            alpha=alpha,
        )
        # plot model fit
        padding = np.ptp(summary.x_center) * x_padding
        x = np.linspace(
            summary.x_center.min() - padding,
            summary.x_center.max() + padding,
            100,
        )
        y = model_fit.params["Intercept"] + model_fit.params["x"] * x
        ax.plot(
            x,
            y,
            color=palette[i],
            lw=0.5,
            ls="--",
        )
        # scale y axis by std
        if var.value.inverse:
            ax.set_ylim(
                data["y"].mean() + y_neg_lim * data["y"].std(),
                data["y"].mean() - y_pos_lim * data["y"].std(),
            )
        else:
            ax.set_ylim(
                data["y"].mean() - y_neg_lim * data["y"].std(),
                data["y"].mean() + y_pos_lim * data["y"].std(),
            )
        ax.set_xlim(
            summary.x_center.min() - 2 * padding,
            summary.x_center.max() + 2 * padding,
        )

        # aes
        axis_side = first_axis_side if not i % 2 else flipside(first_axis_side)
        axis_other_side = flipside(axis_side)
        ax.yaxis.set_label_position(axis_side)
        ax.yaxis.set_ticks_position(axis_side)
        if axis_side == "right":
            ax.spines[axis_side].set_position(("axes", 1 + 0.3 * (i // 2)))
        else:
            ax.spines[axis_side].set_position(("axes", -0.3 * (i // 2)))
        ax.spines[axis_other_side].set_visible(False)
        ax.spines[axis_side].set_visible(True)
        ax.spines[axis_side].set_color(palette[i])
        ax.set_ylabel("corrected " + var.value.name_latex)
        ax.yaxis.label.set_color(palette[i])
        ax.yaxis.set_tick_params(colors=palette[i])

        # add significant stars
        plt.text(
            x=0.1 if axis_side == "left" else 0.9,
            y=0.95 - 0.05 * i // 2,
            s=p_to_star(model_fit.pvalues["x"]),
            ha="left" if i % 2 else "right",
            va="center",
            transform=ax.transAxes,
            color=palette[i],
        )
    plt.sca(ax0)

    if log:
        plt.xlabel(r"$log_{10}$" + "(" + quantity.value.name_latex + ")")
    else:
        plt.xlabel(quantity.value.name_latex)
    fig.tight_layout()

    fit_summary = pd.DataFrame(fit_summary).sort_values(
        "slope_zscore_abs", ascending=False
    )
    return fit_summary


def plot_coef_matrix(
    glmm_rslts_layer,
    variable: Variables,
    quantity: Quantities,
    edge_data: pd.DataFrame,
    legend_size_step=0.5,
    max_size=2.5,
    min_size=0,
):
    # compute std for each variable
    variable_scale = edge_data.loc[:, variable.value.col_name].std()
    variable_scale = -variable_scale if variable.value.inverse else variable_scale

    # scale coef by 1/std
    glmm_rslt = glmm_rslts_layer[(variable.name, quantity.name)]
    emtrends = glmm_rslt.emtrends.set_index("proj_hva_layer")
    meta_info = glmm_rslt.meta_info.set_index("proj_hva_layer")
    emtrends["scaled_coef"] = (
        emtrends[variable.value.col_name + ".trend"] * variable_scale
    )
    if quantity.value.denominator:
        size = (
            meta_info[quantity.value.numerator] / meta_info[quantity.value.denominator]
        ).to_frame("size")
    else:
        size = (meta_info[quantity.value.numerator] / meta_info["n_pairs"]).to_frame(
            "size"
        )
    emtrends = emtrends.join(size)
    emtrends = emtrends.assign(
        pre_hva_layer=lambda df: df.index.str.split("->").str[0],
        post_hva_layer=lambda df: df.index.str.split("->").str[1],
    )
    emtrends = set_dtype(emtrends)
    emtrends = emtrends.set_index(["pre_hva_layer", "post_hva_layer"])

    mat = emtrends.pivot_table(
        index="pre_hva_layer",
        columns="post_hva_layer",
        values=["scaled_coef", "p_adj", "size"],
    )
    # make sure mat is square
    # loop over the first level columns
    for _quantity in mat.columns.levels[0]:
        for _proj in set(mat.index.categories).difference(mat[_quantity].index):
            mat.loc[_proj, :] = np.nan
        for _proj in set(mat.index.categories).difference(mat[_quantity].columns):
            mat.loc[:, (_quantity, _proj)] = np.nan
    mat = mat.sort_index(axis=0).sort_index(axis=1)

    hinton(
        size_matrix=mat["size"].to_numpy(),
        facecolor_matrix=(mat["scaled_coef"]),
        edgecolor_matrix=mat["p_adj"].to_numpy() < 0.05,
        facecolor_cmap="bwr",
        facecolor_cmin=-0.5,
        facecolor_cmax=0.5,
        ticklabels=mat.index.to_series().apply(
            lambda x: x.split("L")[0] + "\nL" + x.split("L")[1]
        ),
        legend_size_step=legend_size_step,
        max_size=max_size,
        min_size=min_size,
        size_as_area=True,
    )
