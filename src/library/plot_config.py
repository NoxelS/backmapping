import os

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
THEMES = ["seaborn-paper", "seaborn-pastel", "seaborn-v0_8-paper", "seaborn-v0_8-pastel"]


def load_matplotlib_local_fonts():
    """
    Load local fonts for matplotlib to use.
    Taken from: https://stackoverflow.com/a/69016300/315168
    """

    font_path = os.path.join("data", "static", "fonts", "Arial.ttf")
    assert os.path.exists(font_path)
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)

    #  Set it as default matplotlib font
    mpl.rc("font", family="sans-serif")
    mpl.rcParams.update(
        {
            "font.sans-serif": prop.get_name(),
        }
    )


def set_plot_config(themes: list = THEMES):
    """
    This function sets the plot configuration for matplotlib
    to make the plots uniform and look like proper paper plots.
    """

    # Load local fonts
    load_matplotlib_local_fonts()

    # Set the theme
    try:
        plt.style.use([theme for theme in plt.style.available if theme in themes][0])
    except IndexError:
        print("No theme found in the available themes. Using default theme.")

    # Set the font size
    mpl.rcParams.update({"font.size": SMALL_SIZE})  # controls default text sizes
    mpl.rcParams.update({"axes.titlesize": 18})  # fontsize of the axes title
    mpl.rcParams.update({"axes.titlepad": 20})  # fontsize of the axes title
    mpl.rcParams.update({"axes.labelsize": MEDIUM_SIZE})  # fontsize of the x and y labels
    mpl.rcParams.update({"xtick.labelsize": SMALL_SIZE})  # fontsize of the tick labels
    mpl.rcParams.update({"ytick.labelsize": SMALL_SIZE})  # fontsize of the tick labels
    mpl.rcParams.update({"legend.fontsize": SMALL_SIZE})  # legend fontsize
    mpl.rcParams.update({"legend.title_fontsize": SMALL_SIZE})
    mpl.rcParams.update({"figure.titlesize": 24})  # fontsize of the figure title
    mpl.rcParams.update({"figure.titleweight": "bold"})  # fontsize of the figure title

    # Set font style
    mpl.rcParams.update({"pdf.fonttype": 42})
    mpl.rcParams.update({"ps.fonttype": 42})
    mpl.rcParams.update({"font.family": "Arial"})

    # Make titles bold
    mpl.rcParams.update({"axes.titleweight": "bold"})
    mpl.rcParams.update({"figure.titleweight": "bold"})

    # Set figure size
    mpl.rcParams.update({"figure.figsize": (10, 6)})

    # Format
    mpl.rcParams.update({"savefig.format": "pdf"})
    mpl.rcParams.update({"savefig.bbox": "tight"})
    mpl.rcParams.update({"savefig.pad_inches": 0.1})

    # Set legend
    mpl.rcParams.update({"legend.frameon": True})
    mpl.rcParams.update({"legend.framealpha": 1})
    mpl.rcParams.update({"legend.shadow": 0})
    mpl.rcParams.update({"legend.fancybox": 1})
    mpl.rcParams.update({"legend.edgecolor": "gray"})
    mpl.rcParams.update({"legend.facecolor": "white"})
    mpl.rcParams.update({"legend.loc": "best"})
    mpl.rcParams.update({"legend.labelspacing": 0.75})
    mpl.rcParams.update({"legend.handletextpad": 0.75})
    mpl.rcParams.update({"legend.handlelength": 1.5})
    mpl.rcParams.update({"legend.borderaxespad": 0.75})

    # Add grid
    mpl.rcParams.update({"axes.grid": True})
    mpl.rcParams.update({"axes.grid.which": "both"})
    mpl.rcParams.update({"grid.linestyle": "--"})
    mpl.rcParams.update({"grid.linewidth": 0.5})
    mpl.rcParams.update({"grid.color": "gray"})
    mpl.rcParams.update({"grid.alpha": 0.5})

    # Set minor and major ticks for grid
    plt.minorticks_on()
    plt.grid(which="major", linestyle="-", linewidth="0.75", color=[0.1, 0.1, 0.1], alpha=0.5)
    plt.grid(which="minor", linestyle="--", linewidth="0.25", color=[0.2, 0.2, 0.2], alpha=0.3)

    # Set line width
    mpl.rcParams.update({"lines.linewidth": 2})

    # Set minor and major ticks
    mpl.rcParams.update({"xtick.major.size": 5})
    mpl.rcParams.update({"xtick.minor.size": 2})
    mpl.rcParams.update({"ytick.major.size": 5})
    mpl.rcParams.update({"ytick.minor.size": 2})
    mpl.rcParams.update({"xtick.direction": "in"})
    mpl.rcParams.update({"ytick.direction": "in"})
    mpl.rcParams.update({"xtick.top": True})
    mpl.rcParams.update({"ytick.right": True})

    # Add space between subplots
    mpl.rcParams.update({"figure.subplot.hspace": 0.9})
