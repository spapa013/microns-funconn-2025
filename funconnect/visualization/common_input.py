import matplotlib.pyplot as plt
import seaborn as sns
from funconnect.utility.plot import ColorPalletes, proj_hva_mapping, p_to_star
from matplotlib.patches import Patch
import pandas as pd
import numpy as np

colormap = {
    "n_synapses": ColorPalletes.Resubmit[0],
    "dend_len": ColorPalletes.Resubmit[1],
    "pred_n_synapses|dend_len": ColorPalletes.Resubmit[3],
    "pred_dend_len": ColorPalletes.Resubmit[4],
    "pred_n_synapses": ColorPalletes.Resubmit[5],
}


def plot_common_input_boxplot(
    stats_result,
    pre_mean_corr,
):
    """
    Plot boxplot with stars
    """
    # input verification
    # verify that only one effect exists in stats_result
    assert len(stats_result.effect.unique()) == 1
    # verify only two weights exist in pre_mean_corr
    assert len(pre_mean_corr.weight.unique()) == 2
    # veiry only one type exists in stats_result and pre_mean_corr
    assert len(stats_result.type.unique()) == 1
    assert len(pre_mean_corr.type.unique()) == 1

    summary = (
        pre_mean_corr.groupby(["proj_hva", "weight", "type"], observed=True)
        .apply(
            lambda df: pd.Series({
                "mean": df["pre_mean"].mean(),
                "sem": df["pre_mean"].sem(),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    effect = stats_result.effect.unique()[0]
    weights = pre_mean_corr["weight"].unique()
    if weights[0].startswith("pred_"):
        expected = weights[0]
        observed = weights[1]
    else:
        expected = weights[1]
        observed = weights[0]

    fig, ax = plt.subplots(1, 1, figsize=(3, 3.5))
    proj_idx = np.arange(pre_mean_corr.proj_hva.nunique())
    bar_width = 0.5
    bar_spacing = 0
    proj_spacing = 0.2
    proj_width = (
        bar_width + bar_spacing
    ) * pre_mean_corr.weight.nunique() + proj_spacing
    bar_positions = []
    for j, w in enumerate(pre_mean_corr["weight"].unique()):
        for k, proj in enumerate(pre_mean_corr["proj_hva"].unique()):
            bar_data = summary.query(f"weight=='{w}' and proj_hva=='{proj}'")
            assert len(bar_data) == 1
            p_data = pre_mean_corr.query(f"weight=='{w}' and proj_hva=='{proj}'")
            bar_pos = k * proj_width + j * (bar_width + bar_spacing)
            # add boxplot first
            bp_stats = ax.boxplot(
                p_data["pre_mean"],
                positions=[bar_pos],
                widths=0.3,
                patch_artist=True,
                boxprops=dict(facecolor="none", edgecolor=colormap[w], linewidth=0.8),
                whiskerprops=dict(color=colormap[w], linewidth=0.8),
                capprops=dict(color=colormap[w], linewidth=0.8),
                medianprops=dict(color=colormap[w], linewidth=0.8),
                flierprops=dict(
                    marker="o",
                    markersize=2,
                    markerfacecolor="none",
                    markeredgecolor=colormap[w],
                    markeredgewidth=0.8,
                ),
                zorder=1,  # lower zorder to put box behind points
            )
            bar_positions.append(
                dict(
                    proj=proj,
                    weight=w,
                    x=bar_pos,
                    y=bar_data["mean"].to_numpy()[0],
                    y_top=bp_stats["whiskers"][1].get_ydata()[1],
                )
            )

    bar_positions = pd.DataFrame(bar_positions)
    ax.set_xticks(
        proj_idx * proj_width
        + bar_width * (pre_mean_corr.weight.nunique() - 1) / 2
        + bar_spacing
    )
    ax.set_xticklabels([proj_hva_mapping[p] for p in pre_mean_corr.proj_hva.unique()])

    # add stars
    y_offset_unit_data = ax.get_ylim()[1] * 0.07
    stats_result = stats_result.copy()
    stats_result["star"] = stats_result["p_adj"].apply(p_to_star)
    for s in stats_result.itertuples():
        # find the bar position
        pos_1 = bar_positions.query(
            f"proj=='{s.proj_hva}' and weight==@observed"
        ).x.to_numpy()
        assert len(pos_1) == 1
        pos_1 = pos_1[0]

        pos_2 = bar_positions.query(
            f"proj=='{s.proj_hva}' and weight==@expected"
        ).x.to_numpy()
        assert len(pos_2) == 1
        pos_2 = pos_2[0]

        y_top = bar_positions.query(f"proj=='{s.proj_hva}'").y_top.max()
        y_offset = y_offset_unit_data
        # add stars
        if s.star == "n.s.":
            ax.text(
                (pos_1 + pos_2) / 2,
                y_top + y_offset,
                str(s.star),
                ha="center",
                va="bottom",
                fontsize=6,
                color="black",
            )
        else:
            ax.text(
                (pos_1 + pos_2) / 2,
                y_top + y_offset,
                str(s.star),
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
            )
        # draw lines
        ax.plot([pos_1, pos_2], [y_top + y_offset] * 2, color="black", lw=0.5)

    # custom legend
    label_mapping = {
        observed: "observed",
        expected: "expected",
    }
    ax.legend(
        handles=[
            Patch(
                color=colormap[p],
                label=label_mapping[p],
            )
            for j, p in enumerate(pre_mean_corr["weight"].unique())
        ],
        loc="upper right",
        bbox_to_anchor=(1, 1),
        frameon=False,
    )

    sns.despine()
    plt.ylabel("Signal correlation")
    # rotate x-axis labels
    plt.xticks(rotation=45, ha="right")
    # title
    plt.title("mean postsyn-postsyn similarity\n" + effect)
