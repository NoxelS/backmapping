import os

import matplotlib.pyplot as plt
import numpy as np

from library.config import Keys, config
from library.datagen.topology import get_ic_from_index, get_ic_type_from_index, ic_to_hlabel
from library.plot_config import set_plot_config
from library.static.utils import inverse_scale_output_ic, to_significant

PATH_TO_HIST = os.path.join(config(Keys.DATA_PATH), "hist")


def plot_hist_single(file: str, plot_name: str = None):
    # Load plot config
    set_plot_config()
    fig, ax = plt.subplots()

    # Load csv
    hist = np.loadtxt(os.path.join(config(Keys.DATA_PATH), "hist", file), delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))

    # Model name and ic index from file name
    model_name = file.replace("training_history_", "").replace(".csv", "")
    ic_index = int(model_name[::-1].split("_", 1)[0][::-1])
    config_name = model_name[::-1].split("_", 1)[1][::-1]
    ic_type = get_ic_type_from_index(ic_index)
    dim = "Å" if ic_type == "bond" else "a.u."

    # If only one row, add dimension
    if len(hist.shape) == 1:
        hist = hist.reshape(1, -1)

    hist_epoch, hist_loss, hist_lr, hist_mae, hist_mse, hist_val_loss, hist_val_mae, hist_val_mse = (
        hist[:, 0],
        hist[:, 1],
        hist[:, 2],
        hist[:, 3],
        hist[:, 4],
        hist[:, 5],
        hist[:, 6],
        hist[:, 7],
    )

    # Scale the losses accordingly
    if ic_type == "bond":
        hist_loss = inverse_scale_output_ic(ic_index, hist_loss) - inverse_scale_output_ic(ic_index, 0)
        hist_val_loss = inverse_scale_output_ic(ic_index, hist_val_loss) - inverse_scale_output_ic(ic_index, 0)

    # There are maybe multiple train cylces so reindex the epochs accordingly
    hist_epoch = np.arange(hist.shape[0])

    # Calculate the minimum loss
    min_loss = np.min(hist_loss)
    min_val_loss = np.min(hist_val_loss)

    # Lr_pices defines the epochs where the lr changed
    lr_pices = [i for i in range(1, len(hist_lr)) if hist_lr[i] != hist_lr[i - 1]]
    lr_pices.insert(0, 0)
    lr_pices.append(len(hist_lr) - 1)

    # Make colormap for the lr
    cmap = plt.cm.plasma
    # Normalize the lr
    norm = plt.Normalize(vmin=min(hist_lr), vmax=max(hist_lr))
    # Make colors for each epoch
    colors = [cmap(norm(i)) for i in hist_lr]

    max_y = max(max(hist_loss), max(hist_val_loss))
    min_y = min(min(hist_loss), min(hist_val_loss))

    # Add h-line for each lr change
    if len(lr_pices) <= 10:
        for i in lr_pices[1:-1]:
            ax.axvline(x=i, color=colors[i], linestyle=":", alpha=0.4, linewidth=1.25)
            ax.text(
                i - 0.1,
                max_y - 0.2 * (max_y - min_y),
                f"lr: {hist_lr[i]:.1e}",
                rotation=90,
                verticalalignment="center",
                horizontalalignment="right",
                fontsize=12,
                color="black",
                fontweight=500,
                alpha=0.75,
            )

    # Plot minimum loss
    ax.axhline(y=min_loss, color="black", linestyle="-", alpha=0.5, linewidth=0.75)
    ax.text(
        1,
        min_loss,
        f"Min. Loss: {min_loss:.4f}{dim if ic_type == 'bond' else ''}",
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=10,
        color="black",
        fontweight=500,
        alpha=0.75,
    )

    # Plot minimum loss
    ax.axhline(y=min_val_loss, color="black", linestyle="--", alpha=0.5, linewidth=0.75)
    ax.text(
        1,
        min_val_loss,
        f"Min. Val. Loss: {min_val_loss:.4f}{dim if ic_type == 'bond' else ''}",
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=10,
        color="black",
        fontweight=500,
        alpha=0.75,
    )

    # Plot the loss and val loss picewise for the lr
    for i in range(len(lr_pices) - 1):
        interval = (lr_pices[i], lr_pices[i + 1] + 1)
        lr = hist_lr[interval[0]]

        # Plot the loss
        ax.plot(
            hist_epoch[interval[0] : interval[1]],
            hist_loss[interval[0] : interval[1]],
            color=colors[lr_pices[i]],
            # marker="s",
            markeredgecolor="black",
            markerfacecolor="white",
            linestyle="-",
            linewidth=1,
        )
        ax.plot(
            hist_epoch[interval[0] : interval[1]],
            hist_val_loss[interval[0] : interval[1]],
            color=colors[lr_pices[i]],
            # marker="v",
            markeredgecolor="black",
            markerfacecolor="white",
            linestyle="--",
            linewidth=1,
        )

    # Make pseudo plot for the legend
    ax.plot(
        [],
        [],
        color="black",
        # marker="s",
        linestyle="-",
        label="Training Loss",
        markerfacecolor="white",
    )
    ax.plot(
        [],
        [],
        color="black",
        # marker="v",
        linestyle="--",
        label="Validation Loss",
        markerfacecolor="white",
    )

    # Add labels
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Loss ({dim})")

    # Plot legend outside of plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    # Plot colorscale for the lr on the right
    ax2 = fig.add_axes([0.94, 0.11, 0.02, 0.57])
    cb1 = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax2, orientation="vertical")
    ax2.set_title("Learning Rate", fontsize=12, color="black", fontweight=600, alpha=0.75, ha="left")
    # ax2.set_yscale("log")

    # Set the y-ticks for the colorbar
    # ax2.set_yticks(hist_lr[lr_pices])
    ax2.get_yaxis().get_major_formatter().labelOnlyBase = False

    ax.get_yaxis().get_major_formatter().labelOnlyBase = False
    ax.set_yscale("log")

    # ax.set_xticks(np.arange(len(hist_epoch)))

    title = f"Training History for {ic_type.capitalize()} {'Angle ' if ic_type == 'dihedral' else ''}{ic_to_hlabel(get_ic_from_index(ic_index))} ({config_name})"
    ax.set_title(title)

    fig.savefig(plot_name if plot_name else f"hist_{model_name}.pdf")


def plot_hist_multiple(files: list, plot_name: str = None, epoch_range: tuple = None, plot_table: bool = False):
    # Load plot config
    set_plot_config(themes=["seaborn-paper"])

    # Add two plots for loss and val loss side by side
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    # Reduce vertical space between plots
    fig.subplots_adjust(hspace=0.5)

    for file in files:
        try:
            # Load csv
            hist = np.loadtxt(os.path.join(config(Keys.DATA_PATH), "hist", file), delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))

            # Model name and ic index from file name
            model_name = file.replace("training_history_", "").replace(".csv", "")
            ic_index = int(model_name[::-1].split("_", 1)[0][::-1])
            config_name = model_name[::-1].split("_", 1)[1][::-1]
            ic_type = get_ic_type_from_index(ic_index)
            dim = "Å" if ic_type == "bond" else "a.u."

            # If only one row, add dimension
            if len(hist.shape) == 1:
                hist = hist.reshape(1, -1)

            lower_range = max(min(hist[:, 0]), epoch_range[0]) if epoch_range else min(hist[:, 0])
            upper_range = min(max(hist[:, 0]), epoch_range[1] + 1) if epoch_range else max(hist[:, 0]) + 1
            hist_range = [int(lower_range), int(upper_range)]

            hist_epoch, hist_loss, hist_lr, hist_mae, hist_mse, hist_val_loss, hist_val_mae, hist_val_mse = (
                hist[hist_range[0] : hist_range[1], 0],
                hist[hist_range[0] : hist_range[1], 1],
                hist[hist_range[0] : hist_range[1], 2],
                hist[hist_range[0] : hist_range[1], 3],
                hist[hist_range[0] : hist_range[1], 4],
                hist[hist_range[0] : hist_range[1], 5],
                hist[hist_range[0] : hist_range[1], 6],
                hist[hist_range[0] : hist_range[1], 7],
            )

            # Scale the losses accordingly
            if ic_type == "bond":
                hist_loss = inverse_scale_output_ic(ic_index, hist_loss) - inverse_scale_output_ic(ic_index, 0)
                hist_val_loss = inverse_scale_output_ic(ic_index, hist_val_loss) - inverse_scale_output_ic(ic_index, 0)

            # There are maybe multiple train cylces so reindex the epochs accordingly
            hist_epoch = np.arange(hist.shape[0]) if not epoch_range else np.arange(hist_range[0], hist_range[1])

            # Calculate the minimum loss
            min_loss = np.min(hist_loss)
            min_val_loss = np.min(hist_val_loss)

            # Plot the loss on the left
            axs[0].plot(
                hist_epoch,
                hist_loss,
                label=model_name,
                markeredgecolor="black",
                markerfacecolor="white",
                linestyle="-",
                linewidth=1.2,
                alpha=0.75,
            )

            # Plot the val loss on the right
            axs[1].plot(
                hist_epoch,
                hist_val_loss,
                label=model_name,
                markeredgecolor="black",
                markerfacecolor="white",
                linestyle="-",
                linewidth=1.2,
                alpha=0.75,
            )
        except Exception as e:
            print(f"Error: {e}")

    # Add labels
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel(f"Loss ({dim})")

    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel(f"Val. Loss ({dim})")

    # Set the y-axis to log scale
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")

    # Set positions of axes so its always the same
    axs[1].set_position([0.1, 0, 0.8, 0.4])
    axs[0].set_position([0.1, 0.55, 0.8, 0.4])

    if len(files) < 31 * 2:
        # Plot one legend beneath the others for both plots outside of the plot
        axs[0].legend(bbox_to_anchor=(1.05, 0.9), loc="upper left", borderaxespad=0.0, ncol=2)

        # Add title to legend
        axs[0].text(
            0.945,
            0.95,
            "Models",
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=12,
            color="black",
            fontweight=600,
            transform=fig.transFigure,
        )

    # Plot a table with every hist file and the corresponding min loss and val loss
    # this plot should be beneath the legend
    if plot_table:
        rows = []
        for file in files:
            try:
                # Load csv
                hist = np.loadtxt(os.path.join(config(Keys.DATA_PATH), "hist", file), delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))

                # Model name and ic index from file name
                model_name = file.replace("training_history_", "").replace(".csv", "")
                ic_index = int(model_name[::-1].split("_", 1)[0][::-1])
                config_name = model_name[::-1].split("_", 1)[::-1]
                ic_type = get_ic_type_from_index(ic_index)
                dim = "Å" if ic_type == "bond" else "a.u."

                # If only one row, add dimension
                if len(hist.shape) == 1:
                    hist = hist.reshape(1, -1)

                hist_epoch, hist_loss, hist_lr, hist_mae, hist_mse, hist_val_loss, hist_val_mae, hist_val_mse = (
                    hist[hist_range[0] : hist_range[1], 0],
                    hist[hist_range[0] : hist_range[1], 1],
                    hist[hist_range[0] : hist_range[1], 2],
                    hist[hist_range[0] : hist_range[1], 3],
                    hist[hist_range[0] : hist_range[1], 4],
                    hist[hist_range[0] : hist_range[1], 5],
                    hist[hist_range[0] : hist_range[1], 6],
                    hist[hist_range[0] : hist_range[1], 7],
                )

                # Calculate the minimum loss
                min_loss = np.min(hist_loss)
                min_val_loss = np.min(hist_val_loss)

                rows.append([model_name, to_significant(min_loss, 5), to_significant(min_val_loss, 5)])
            except Exception as e:
                print(f"Error: {e}")

        # Sort by min val loss
        rows = sorted(rows, key=lambda x: x[2])

        # Plot the table
        height = 1 / 25 * (len(rows))
        axs[1].table(
            cellText=rows,
            colLabels=["Model", "Min. Loss (a.u.)", "Min. Val. Loss (a.u.)"],
            loc="bottom",
            bbox=[0.94, 0, 0.45, height],
            cellLoc="left",
            colLoc="left",
            transform=fig.transFigure,
        )

        # Add title to table
        axs[1].text(
            0.945,
            height + 0.04,
            "Model Comparison",
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=12,
            color="black",
            fontweight=600,
            # transform axes to global
            transform=fig.transFigure,
        )

    # Add title
    axs[0].set_title("Training Loss")

    # Add title
    axs[1].set_title("Validation Loss")

    fig.savefig(plot_name if plot_name else f"hist_{model_name}.pdf")


def plot_diff_multiple(files: list, plot_name: str = None, epoch_range: tuple = None, plot_table: bool = False, small: bool = False):
    # Load plot config
    set_plot_config(themes=["seaborn-paper"])

    # Add one plot for the diff
    fig, ax = plt.subplots()

    average = []
    longest_hist = []
    max_y = 0
    min_y = 0
    plotted_files = 0

    def color_from_loss(loss):
        if loss < 0:
            return "purple"
        elif loss > 0:
            return "red"
        else:
            return "green"

    for file in files:
        try:
            # Load csv
            hist = np.loadtxt(
                os.path.join(config(Keys.DATA_PATH), "hist", file),
                delimiter=",",
                skiprows=1,
                usecols=(0, 1, 2, 3, 4, 5, 6, 7),
            )

            # Model name and ic index from file name
            model_name = file.replace("training_history_", "").replace(".csv", "")
            ic_index = int(model_name[::-1].split("_", 1)[0][::-1])
            config_name = model_name[::-1].split("_", 1)[1][::-1]
            ic_type = get_ic_type_from_index(ic_index)
            dim = "Å" if ic_type == "bond" else "a.u."

            # If only one row, add dimension
            if len(hist.shape) == 1:
                hist = hist.reshape(1, -1)

            lower_range = max(min(hist[:, 0]), epoch_range[0]) if epoch_range else min(hist[:, 0])
            upper_range = min(max(hist[:, 0]), epoch_range[1] + 1) if epoch_range else max(hist[:, 0]) + 1
            hist_range = [int(lower_range), int(upper_range)]

            (
                hist_epoch,
                hist_loss,
                hist_lr,
                hist_mae,
                hist_mse,
                hist_val_loss,
                hist_val_mae,
                hist_val_mse,
            ) = (
                hist[hist_range[0] : hist_range[1], 0],
                hist[hist_range[0] : hist_range[1], 1],
                hist[hist_range[0] : hist_range[1], 2],
                hist[hist_range[0] : hist_range[1], 3],
                hist[hist_range[0] : hist_range[1], 4],
                hist[hist_range[0] : hist_range[1], 5],
                hist[hist_range[0] : hist_range[1], 6],
                hist[hist_range[0] : hist_range[1], 7],
            )

            # Scale the losses accordingly
            if ic_type == "bond":
                hist_loss = inverse_scale_output_ic(ic_index, hist_loss) - inverse_scale_output_ic(ic_index, 0)
                hist_val_loss = inverse_scale_output_ic(ic_index, hist_val_loss) - inverse_scale_output_ic(ic_index, 0)

            # There are maybe multiple train cylces so reindex the epochs accordingly
            hist_epoch = np.arange(hist.shape[0]) if not epoch_range else np.arange(hist_range[0], hist_range[1])

            # Calculate the diff
            diff = hist_loss - hist_val_loss

            # Add average
            average.append(diff)

            # Set min and max
            max_y = max(max_y, max(diff))
            min_y = min(min_y, min(diff))
            longest_hist = hist_epoch if len(hist_epoch) > len(longest_hist) else longest_hist

            # Plot the loss on the left
            ax.plot(
                hist_epoch,
                diff,
                label=model_name.upper(),
                markeredgecolor="black",
                markerfacecolor="white",
                linestyle="-",
                linewidth=1.2,
                alpha=0.75,
            )

            plotted_files += 1
        except Exception as e:
            print(f"Error: {e}")

    # Build average with different hist sizes
    average_values = np.zeros(len(longest_hist))
    for i in range(len(longest_hist)):
        average_values[i] = np.mean([a[i] for a in average if i < len(a)])

    # Plot the average
    ax.plot(
        longest_hist,
        average_values,
        label="Average",
        markeredgecolor="black",
        markerfacecolor="white",
        color="black",
        linestyle="-",
        linewidth=1.2,
        alpha=1,
    )

    # Add labels
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\mathcal{L}_{train} - \mathcal{L}_{val}$" + f" (a.u.)")

    # Set the y-axis to log scale
    # ax.set_xscale("log")
    # ax.set_yscale("log")

    if plotted_files < 18 * 2 and not small:
        # Plot one legend beneath the others for both plots outside of the plot
        ax.legend(bbox_to_anchor=(1.05, 0.9), loc="upper left", borderaxespad=0.0, ncol=2)

        # Add title to legend
        ax.text(
            0.945,
            0.85,
            "Models",
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=12,
            color="black",
            fontweight=600,
            transform=fig.transFigure,
        )

    # Add title
    ax.set_title("Loss Differences") if not small else None

    # If small, increase font sizeso of axis
    if small:
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.tick_params(axis="both", which="minor", labelsize=8)

        # Set labels
        ax.set_xlabel("Epoch", fontsize=18)
        ax.set_ylabel(r"$\mathcal{L}_{train} - \mathcal{L}_{val}$" + f" ({dim})", fontsize=18)

    fig.savefig(plot_name if plot_name else f"hist_{model_name}_diff.pdf", transparent=small)
