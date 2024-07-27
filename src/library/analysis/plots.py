import configparser
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew

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


def plot_hist_multiple(files: list, plot_name: str = None, epoch_range: tuple = None, plot_table: bool = False, parameter_name: str = None):
    # Load plot config
    set_plot_config(themes=["seaborn-paper"])

    # Configs
    configs = {}

    # Go thorugh all config files and map the file name to the model name
    for config_name in [f for f in os.listdir(os.path.join("data", "configs")) if f.endswith(".ini")]:
        # Read the config file with configparser
        with open(os.path.join("data", "configs", config_name), "r") as f:
            config_parser = configparser.ConfigParser()
            config_parser.read_file(f)

            # Get the model name
            model_name = config_parser["INFO"]["model_name_prefix"]

            configs[model_name] = config_parser

    # Add two plots for loss and val loss side by side
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    # Reduce vertical space between plots
    fig.subplots_adjust(hspace=0.5)

    train_loss_lines = []
    val_loss_lines = []

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

            if parameter_name:
                config_for_run = configs[config_name]

                # Calcualte the variable parameter
                # go thorugh all sections
                parameter = None
                for section in config_for_run.sections():
                    # Check if the parameter is in the section
                    if parameter_name in config_for_run[section]:
                        parameter = config_for_run[section][parameter_name]
                        # Remove everything after ; if there is a comment
                        parameter = parameter.split(";")[0].strip()
                        parameter = float(parameter) if "." in parameter else int(parameter)

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
            train_loss = axs[0].plot(
                hist_epoch,
                hist_loss,
                label=model_name if not parameter_name else parameter,
                markeredgecolor="black",
                markerfacecolor="white",
                linestyle="-",
                linewidth=1.2,
                alpha=0.75,
            )

            # Plot the val loss on the right
            val_loss = axs[1].plot(
                hist_epoch,
                hist_val_loss,
                label=model_name if not parameter_name else parameter,
                markeredgecolor="black",
                markerfacecolor="white",
                linestyle="-",
                linewidth=1.2,
                alpha=0.75,
            )

            if parameter_name:
                train_loss_lines.append((parameter, train_loss))
                val_loss_lines.append((parameter, val_loss))

        except Exception as e:
            print(f"Error: {e}")

    if parameter_name:
        skewness = skew([p[0] for p in train_loss_lines])
        use_log_colors = skewness > 1
        transform_func = np.log if use_log_colors else lambda x: x

        # Change the color of the lines based on the parameter
        max_parameter = transform_func(max([p[0] for p in train_loss_lines]))
        min_parameter = transform_func(min([p[0] for p in train_loss_lines]))

        # Fix min parameter if it is 0
        if use_log_colors and min([p[0] for p in train_loss_lines]) == 0:
            min_parameter = transform_func(min([p[0] for p in train_loss_lines if p[0] > 0]))

        # Normalize the parameter and make logarithmic
        norm = plt.Normalize(vmin=min_parameter, vmax=max_parameter)

        # Make colormap for the parameter
        cmap = plt.cm.plasma

        # Make colors for each parameter logarithmically
        colors = [cmap(norm(transform_func(p[0]))) for p in train_loss_lines]

        # Change the color of the lines
        for i, line in enumerate(train_loss_lines):
            line[1][0].set_color(colors[i])
            val_loss_lines[i][1][0].set_color(colors[i])

        # # Plot the colorbar
        # ax2 = fig.add_axes([0.94, 0.01, 0.1, 0.04])
        # cb1 = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax2, orientation="horizontal")
        # ax2.set_title(parameter_name.replace("_", " ").title(), fontsize=12, color="black", fontweight=600, alpha=0.75, ha="left")

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
        if parameter_name:
            handles, labels = axs[0].get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: float(t[0])))
            axs[0].legend(handles, labels, bbox_to_anchor=(1.05, 0.9), loc="upper left", borderaxespad=0.0, ncol=2)
        else:
            handles, labels = axs[0].get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: int("".join(c for c in t[0] if c.isdigit()))))
            # Plot one legend beneath the others for both plots outside of the plot
            axs[0].legend(handles, labels, bbox_to_anchor=(1.05, 0.9), loc="upper left", borderaxespad=0.0, ncol=2)

        # Add title to legend
        axs[0].text(
            0.945,
            0.95,
            "Models" if not parameter_name else parameter_name.replace("_", " ").title(),
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=12,
            color="black",
            fontweight=600,
            transform=fig.transFigure,
        )

    # Plot a table with every hist file and the corresponding min loss and val loss
    # this plot should be beneath the legend
    if plot_table and False:
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


def plot_loss_multiple(files: list, plot_name: str = None, parameter_name: str = None):
    # Load plot config
    set_plot_config(themes=["seaborn-paper"])

    # Add two plots for loss and val loss side by side
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))

    # Reduce vertical space between plots
    fig.subplots_adjust(hspace=0.5)

    # Configs
    configs = {}

    # Store data
    data_train = []
    data_val = []
    data_diff = []

    # Go thorugh all config files and map the file name to the model name
    for config_name in [f for f in os.listdir(os.path.join("data", "configs")) if f.endswith(".ini")]:
        # Read the config file with configparser
        with open(os.path.join("data", "configs", config_name), "r") as f:
            config_parser = configparser.ConfigParser()
            config_parser.read_file(f)

            # Get the model name
            model_name = config_parser["INFO"]["model_name_prefix"]

            configs[model_name] = config_parser

    for file in files:
        try:
            # Load csv
            hist = np.loadtxt(os.path.join(config(Keys.DATA_PATH), "hist", file), delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))

            # Model name and ic index from file name
            model_name = file.replace("training_history_", "").replace(".csv", "")
            ic_index = int(model_name[::-1].split("_", 1)[0][::-1])
            config_name = model_name[::-1].split("_", 1)[1][::-1]
            ic_type = get_ic_type_from_index(ic_index)
            config_for_run = configs[config_name]
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

            # Calcualte the variable parameter
            # go thorugh all sections
            parameter = None
            for section in config_for_run.sections():
                # Check if the parameter is in the section
                if parameter_name in config_for_run[section]:
                    parameter = config_for_run[section][parameter_name]
                    # Remove everything after ; if there is a comment
                    parameter = parameter.split(";")[0].strip()
                    parameter = float(parameter) if "." in parameter else int(parameter)

            # Store the data
            data_train.append((parameter, min_loss))
            data_val.append((parameter, min_val_loss))
            data_diff.append((parameter, min_loss - min_val_loss))

        except Exception as e:
            print(f"Error: {e}")

    # Select min for val and train loss
    min_train = min([d[1] for d in data_train])
    min_val = min([d[1] for d in data_val])

    x_min_train = [d[0] for d in data_train if d[1] == min_train][0]
    x_min_val = [d[0] for d in data_val if d[1] == min_val][0]

    parameter_h = parameter_name.replace("_", " ")

    # Plot the data as scatter plot in one plot
    axs.scatter(*zip(*data_train), label=f"Training Loss", marker="o", color="xkcd:purple")
    axs.scatter(*zip(*data_val), label=f"Validation Loss", marker="v", color="xkcd:blue")

    # Add the minima to the plot
    axs.axhline(min_train, color="xkcd:purple", linestyle="-", alpha=0.5, linewidth=0.75, label=f"Min. Train Loss at {parameter_h.title()} = {x_min_train}")
    axs.axhline(min_val, color="xkcd:blue", linestyle="--", alpha=0.5, linewidth=0.75, label=f"Min. Val. Loss at {parameter_h.title()} = {x_min_val}")

    # Add labels
    axs.set_xlabel(parameter_h.capitalize())
    axs.set_ylabel(f"Loss ({dim})")

    # Add title
    axs.set_title(f"Loss for {parameter_h.title()}")

    # Add legend
    axs.legend()

    # Save the plot
    fig.savefig(plot_name if plot_name else f"hist_{model_name}_lp.pdf")

    # Make the y-axis log scale
    axs.set_yscale("log")
    axs.set_xscale("log")

    # Save the plot again
    fig.savefig(plot_name.replace(".pdf", "_log.pdf") if plot_name else f"hist_{model_name}_lp_log.pdf")
