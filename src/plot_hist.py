import argparse
import logging
import os

from library.config import Keys, config

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

    # Add argument for verbose
    parser.add_argument("-v", "--verbose", action="store_true", help="Turn on verbose output, defaults to off.", default=0)

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

    # Plot the selected files if multiple are selected
    if len(target_files) > 1:
        # Import here to reduce import overhead
        from library.analysis.plots import plot_hist_multiple

        # Plot the selected files
        plot_hist_multiple(target_files, args.plot_name, epoch_range=args.epoch_range)
