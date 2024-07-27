import argparse
import logging
import os

from library.config import Keys, config
from library.datagen.topology import get_ic_type_from_index

if __name__ == "__main__":
    # Select the available hist files
    hist_files = [file for file in os.listdir(os.path.join(config(Keys.DATA_PATH), "hist")) if ".csv" in file]
    hist_names = [file.replace("training_history_", "").replace(".csv", "") for file in hist_files]
    ic_indices = set([int(name[::-1].split("_", 1)[0][::-1]) for name in hist_names])
    config_names = set([name[::-1].split("_", 1)[1][::-1] for name in hist_names])

    # Add argument parser
    parser = argparse.ArgumentParser(description=f"Plot the training history of a model. Hist files are stored in {config(Keys.DATA_PATH)}/hist.")

    # Add internal coordinate index argument
    parser.add_argument("--hist-file", "-f", type=str, help="Select one of the available hist files to plot.")

    # Add argument for multiple hist files
    parser.add_argument("--hist-files", "-fs", type=str, nargs="+", help="Select multiple hist files to plot.")

    # Add internal coordinate index argument
    parser.add_argument("--ic_index", "-i", type=int, help="Select an ic index to plot all hist files for that index.", choices=list(ic_indices))

    # Add internal coordinate index argument
    parser.add_argument("--config-match", "-c", type=str, help="Selects all configs that match for a given substring.")

    # Add argument for filter by ic type
    parser.add_argument("--filter-ic-type", type=str, help="Filter the internal coordinate type to plot.", choices=["angle", "bond", "dihedral"])

    # Add argument for verbose
    parser.add_argument("-v", "--verbose", action="store_true", help="Turn on verbose output, defaults to off.", default=False)

    # Add argument for table
    parser.add_argument("-t", "--table", action="store_true", help="Plot a table with the losses, defaults to off.", default=False)

    # Add argument for table
    parser.add_argument("-lp", "--loss-plot", type=str, help="Plot a plot with min_loss(lp)")

    # Add argument for diff plot
    parser.add_argument("-d", "--plot-diff", action="store_true", help="Add a diff plot for the specified arguments.", default=False)

    # Add argument for small version
    parser.add_argument("-s", "--small", action="store_true", help="Plots the image to be a small plot.", default=False)

    # Add argument for plot name
    parser.add_argument("--plot-name", "-n", type=str, help="Name of the plot to save.")

    # Add argument for custom epoch range
    parser.add_argument("--epoch-range", "-e", type=int, nargs=2, help="Custom epoch range to plot.")

    # Parse the arguments
    args = parser.parse_args()

    # Set up logger with the right verbosity
    verbosity = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format="[%(asctime)s] %(levelname)s {%(name)s:%(filename)s:%(lineno)d}: %(message)s", datefmt="%m-%d %H:%M:%S", level=verbosity)
    logging.debug("Verbose output enabled.") if args.verbose else None

    # Disable matplotlib logging
    if not args.verbose:
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("matplotlib.pyplot").setLevel(logging.ERROR)
        logging.getLogger("fontTools").setLevel(logging.ERROR)
        logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
        logging.getLogger("numexpr.utils").setLevel(logging.ERROR)

    target_files = []

    # Check if hist file is provided
    if args.hist_file:
        target_files.extend([file for file in hist_files if args.hist_file == file])

    # Check if hist files are provided
    if args.hist_files:
        target_files.extend([file for file in hist_files if file in args.hist_files])

    # Check if ic index is provided
    if args.ic_index:
        target_files.extend([file for file in hist_files if f"_{args.ic_index}." in file])

    # Check if config match is provided
    if args.config_match:
        target_files.extend([file for file in hist_files if args.config_match in file])

    if args.filter_ic_type:
        # Filter the target files
        target_files = [file for file in target_files if get_ic_type_from_index(int(file.split(".csv")[0][::-1].split("_", 1)[0][::-1])) == args.filter_ic_type]

    # Check if no arguments are provided
    if not args.hist_file and not args.hist_files and not args.ic_index and not args.config_match:
        exit("Please provide at least one argument.")

    if not target_files:
        exit("No files found for the given arguments.")

    # Log the selected files
    logging.info(f"Selected files ({len(target_files)}):")
    for file in target_files:
        lines_in_file = sum(1 for line in open(os.path.join(config(Keys.DATA_PATH), "hist", file)))
        logging.info(f" - {file} ({lines_in_file-2} epochs)")

    # Check if any file is empty and log it
    for file in target_files:
        if os.path.getsize(os.path.join(config(Keys.DATA_PATH), "hist", file)) == 0:
            logging.warning(f"File {file} is empty.")

            # Remove from target files
            target_files.remove(file)

            if len(target_files) == 1:
                exit("Selected file is empty.")

    # Plot the selected file if only one is selected
    if len(target_files) == 1:
        # Import here to reduce import overhead
        from library.analysis.plots import plot_hist_single

        # Plot the selected file
        plot_hist_single(target_files[0], args.plot_name)

        if args.loss_plot:
            logging.info("Loss plot requires more than one file to be selected.")

    # Plot the selected files if multiple are selected
    if len(target_files) > 1:
        # Import here to reduce import overhead
        from library.analysis.plots import plot_diff_multiple, plot_hist_multiple

        # Plot the selected files
        plot_hist_multiple(target_files, args.plot_name, epoch_range=args.epoch_range, plot_table=args.table, parameter_name=args.loss_plot)
        # Plot diff if selected
        (
            plot_diff_multiple(
                target_files, args.plot_name.replace(".", "_diff.") if args.plot_name else None, epoch_range=args.epoch_range, plot_table=args.table, small=args.small
            )
            if args.plot_diff
            else None
        )

        if args.loss_plot:
            from library.analysis.plots import plot_loss_multiple

            plot_loss_multiple(files=target_files, plot_name=args.plot_name.replace(".", "_lp.") if args.plot_name else None, parameter_name=args.loss_plot)
