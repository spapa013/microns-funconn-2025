import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from funconnect.utility.pandas_util import CATEGORICAL
from seaborn import axes_style

proj_hva_mapping = {
    "V1->V1": r"$V1 \rightarrow V1$",
    "HVA->HVA": r"$HVA \rightarrow HVA$",
    "V1->HVA": r"$V1 \rightarrow HVA$",
    "HVA->V1": r"$HVA \rightarrow V1$",
}


def latex_arrow(string):
    return r"{}".format(string.replace("->", "$\rightarrow$"))


def proj_to_latex(proj):
    return r"${}$".format(proj.replace("->", " \\rightarrow "))


proj_hva_layer_mapping = {
    c: proj_to_latex(c) for c in CATEGORICAL["proj_hva_layer"].categories
}


class ColorPalletes:
    BioRxiv = ["#000000", "#D00000", "#3F88C5"]
    BioRxivDesat = ["#505253", "#E06268", "#77ADD8"]
    Desat4 = ["#505253", "#A64C62", "#E06268", "#77ADD8"]
    CTB = ["#f97171", "#8ad6cc", "#385a7c"]
    C3 = ["#4D5AA3", "#E89C0E", "k"]
    K = ["k"]
    # Resubmit = ["#505253", "#E06268", "#6D84A1", "#A64C62", "#956ba3", "#3c5985"]
    Resubmit = ["#505253", "#E06268", "#77ADD8", "#A64C62", "#c09acd", "#6D84A1"]


class rcParams:
    arial = {
        **axes_style("ticks"),
        # font
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "mathtext.it": "Arial:italic",
        "mathtext.cal": "Arial:italic",
        "mathtext.default": "regular",
        "mathtext.fontset": "custom",
        "font.size": 6,
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 5,
        "legend.title_fontsize": 6,
        # despine
        "axes.spines.top": False,
        "axes.spines.right": False,
        # dpi
        "figure.dpi": 300,
        # grid
        "axes.grid": False,
        # default color cycle
        "axes.prop_cycle": plt.cycler("color", ColorPalletes.BioRxiv),
        # scater size
        "scatter.marker": "o",
        "lines.markersize": 1.5,
        "lines.linewidth": 0.8,
        # editable text
        "text.usetex": False,
        # have ticks on the edge
        "axes.autolimit_mode": "round_numbers",
    }
    arial_desat = {
        **arial,
        "axes.prop_cycle": plt.cycler("color", ColorPalletes.BioRxivDesat),
    }
    arial_desat4 = {
        **arial,
        "axes.prop_cycle": plt.cycler("color", ColorPalletes.Desat4),
    }


def get_xticks(ax=None, in_lim=True):
    ax = ax or plt.gca()
    xtick, xticklabels = ax.get_xticks(), ax.get_xticklabels()
    if in_lim:
        xlim = ax.get_xlim()
        xmin = min(xlim)
        xmax = max(xlim)
        in_lim = (xtick >= xmin) & (xtick <= xmax)
        xtick, xticklabels = (
            xtick[in_lim],
            [t for t, _in in zip(xticklabels, in_lim) if _in],
        )
    return xtick, xticklabels


def get_yticks(ax=None, in_lim=True):
    ax = ax or plt.gca()
    ytick, yticklabels = ax.get_yticks(), ax.get_yticklabels()
    if in_lim:
        ylim = ax.get_ylim()
        ymin = min(ylim)
        ymax = max(ylim)
        in_lim = (ytick >= ymin) & (ytick <= ymax)
        ytick, yticklabels = (
            ytick[in_lim],
            [t for t, _in in zip(yticklabels, in_lim) if _in],
        )
    return ytick, yticklabels


def min_max_labels(ax=None, x=True, y=True):
    ax = ax or plt.gca()

    if x:
        x_tick, x_ticklabels = get_xticks(ax)
        x_ticklabels[1:-1] = ["" for _ in x_ticklabels[1:-1]]
        ax.set_xticks(x_tick)
        ax.set_xticklabels(x_ticklabels)

    if y:
        y_tick, y_ticklabels = get_yticks(ax)
        y_ticklabels[1:-1] = ["" for _ in y_ticklabels[1:-1]]
        ax.set_yticks(y_tick)
        ax.set_yticklabels(y_ticklabels)


def p_to_star(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


def inch_to_data_distance(ax, inch):
    # get transform from inches to display
    fig = ax.get_figure()
    inch2display = fig.dpi_scale_trans
    # get transfrom from display to data
    display2data = ax.transData.inverted()
    coor0 = [0, 0]  # inch
    coor1 = inch
    data0 = display2data.transform(inch2display.transform(coor0))
    data1 = display2data.transform(inch2display.transform(coor1))
    return data1 - data0


def normalized_cmap(array, cmap, cmin=None, cmax=None):
    cmin = array.min() if cmin is None else cmin
    cmax = array.max() if cmax is None else cmax
    if cmin > cmax:
        array = -array
        cmin, cmax = -cmin, -cmax
    array = colors.Normalize(vmin=cmin, vmax=cmax)(array)
    cmap = cm.get_cmap(cmap)
    array = cmap(array)
    return array


def hinton(
    size_matrix=None,
    facecolor_matrix=None,
    edgecolor_matrix=None,
    annotate_matrix=None,
    ticklabels=None,
    xlabel=None,
    ylabel=None,
    max_size=None,
    min_size=None,
    legend_size_step=50,
    facecolor_cmap="bwr",
    facecolor_cmin=None,
    facecolor_cmax=None,
    edgecolor_cmap="Greys",
    edgecolor_cmin=0,
    edgecolor_cmax=1,
    legend=True,
    size_as_area=False,
):
    # check arguments
    for m in (size_matrix, facecolor_matrix, edgecolor_matrix):
        try:
            matrix_shape = m.shape
            break
        except AttributeError:
            continue
    else:
        raise ValueError("At least one input matrix should be not None.")
    size_matrix = (
        np.ones(matrix_shape) if size_matrix is None else np.nan_to_num(size_matrix)
    )
    facecolor_matrix = (
        np.zeros(matrix_shape)
        if facecolor_matrix is None
        else np.nan_to_num(facecolor_matrix)
    )
    edgecolor_matrix = (
        np.zeros(matrix_shape)
        if edgecolor_matrix is None
        else np.nan_to_num(edgecolor_matrix)
    )
    assert size_matrix.shape == facecolor_matrix.shape == edgecolor_matrix.shape
    if not max_size:
        max_size = size_matrix.max()
    if not facecolor_cmin:
        facecolor_cmin = -np.nanmax(np.abs(facecolor_matrix))
    if not facecolor_cmax:
        facecolor_cmax = np.nanmax(np.abs(facecolor_matrix))

    # preprocess arguments
    edgecolor_matrix = normalized_cmap(
        edgecolor_matrix, edgecolor_cmap, edgecolor_cmin, edgecolor_cmax
    )
    facecolor_matrix = normalized_cmap(
        facecolor_matrix, facecolor_cmap, facecolor_cmin, facecolor_cmax
    )
    size_matrix = (
        np.clip(size_matrix, a_min=min_size, a_max=max_size) / max_size
    )  # clip and normalize to [0,1]
    if size_as_area:
        size_matrix = np.sqrt(size_matrix)
    size_scale = 0.75
    size_matrix *= size_scale  # scale for padding

    x, y = np.meshgrid(np.arange(size_matrix.shape[1]), np.arange(size_matrix.shape[0]))

    # create axes for plotting, axes include:
    # 1. ax: main axis for plotting the hinton matrix
    # 2. cax: axis for colorbar
    fig = plt.figure(figsize=[4, 4], layout="tight")
    n_col = (len(x) + 1) * 2
    ax = plt.subplot2grid((n_col - 1, n_col), (0, 0), colspan=n_col, rowspan=n_col - 2)
    cax = plt.subplot2grid(
        (n_col - 1, n_col), (n_col - 2, 1), colspan=n_col - 5, rowspan=1
    )

    # generate patches
    patches = [
        Rectangle([xi - s / 2, yi - s / 2], s, s)
        for xi, yi, s in zip(x.ravel(), y.ravel(), size_matrix.ravel())
    ]
    patches = PatchCollection(
        patches,
        facecolors=facecolor_matrix.reshape(-1, 4),
        edgecolors=edgecolor_matrix.reshape(-1, 4),
        linewidth=1.5,
    )

    # plot
    plt.sca(ax)
    ax.set_aspect("equal", "box")
    # add background
    background = Rectangle(
        [0 - 0.5, 0 - 0.5],
        size_matrix.shape[0],
        size_matrix.shape[1],
        facecolor="grey",
        edgecolor="k",
        linewidth=2,
    )
    ax.add_patch(background)
    ## add square grid
    ax.add_collection(patches)
    ax.spines[["right", "top", "left", "bottom"]].set_visible(True)
    if annotate_matrix is not None:
        for xi, yi, ai in zip(x.ravel(), y.ravel(), annotate_matrix.ravel()):
            ax.annotate(ai, (xi, yi), ha="center", va="center")

    # create legends
    if legend:
        # size legend
        size_step = legend_size_step / max_size
        padding = 0.1
        y_legend = y / 4 * 5
        size_patches = [
            Rectangle([xi + padding - s / 2, yi - s / 2], s, s)
            for xi, yi, s in zip(
                x[:, -1].ravel() + 1,
                y_legend[:, -1].ravel(),
                np.sqrt(np.arange(size_step, 1.1, size_step)) * size_scale
                if size_as_area
                else np.arange(size_step, 1.1, size_step) * size_scale,
            )
        ]
        size_patches = PatchCollection(size_patches, color="k", linewidth=2)
        size_legend = np.arange(legend_size_step, max_size + 1, legend_size_step)
        for xi, yi, s in zip(x[:, -1], y_legend[:, -1], size_legend):
            ax.annotate(s, (xi + 2 + padding, yi), ha="center", va="center")
        ax.add_collection(size_patches)

        # color legend
        cmap = mpl.colormaps[facecolor_cmap]
        norm = mpl.colors.Normalize(vmin=facecolor_cmin, vmax=facecolor_cmax)
        cb = mpl.colorbar.Colorbar(
            cax,
            cmap=cmap,
            norm=norm,
            ticks=np.linspace(facecolor_cmin, facecolor_cmax, 5),
            orientation="horizontal",
        )

    plt.sca(ax)
    plt.xlim([0 - 0.6, size_matrix.shape[1] + 2 - 0.4])
    plt.ylim([0 - 0.6, size_matrix.shape[0] - 0.4])
    ax.invert_yaxis()
    sns.despine()
    if ticklabels is None:
        ticklabels = np.arange(len(size_matrix))
    plt.xticks(np.arange(len(ticklabels)), ticklabels)
    plt.yticks(np.arange(len(ticklabels)), ticklabels)
    ax.tick_params(left=False, bottom=False)
    [s.set_visible(False) for s in ax.spines.values()]
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    return fig


# Example plotting code:
# fig = plt.figure(figsize=(6, 4), dpi=300)
# plot = sns.barplot(
#     data=bs_df,
#     x="mask_overlap_bin",
#     y="readout_similarity_cvt",
#     hue="population",
#     errorbar=("pi", 95),
#     palette=ColorPalletes.BioRxiv[::-1],
#     alpha=0.5,
#     saturation=1,
#     gap=.05,
#     err_kws=dict(lw=1.5),
# )
# for patch in plt.gca().patches:
#     clr = patch.get_facecolor()
#     patch.set_edgecolor(clr)
#     patch.set_linewidth(1.5)
