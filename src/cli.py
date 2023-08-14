import argparse
from library import __version__
from library.viz import show_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f'CLI for the backmapping CNN library. Version: {__version__.v}')
    
    # Version
    parser.add_argument('--version', action='version', version=__version__.v)

    # Main function to run
    parser.add_argument('cmd', type=str, default='', help='The main function to run.',
                        choices=["show"])

    parser.add_argument('options', type=str, nargs='*', help='Options for the main function to run.')

    args = parser.parse_args()

    # Run the main function
    if args.cmd == 'show':
        """
            Show a dataset with the given name and residue index in a 3D plot.
        """
        if len(args.options) != 2:
            raise ValueError('Show requires 2 arguments. E.g cli.py show CG2AT_2023-02-13_20-20-52 2')

        # Get the arguments
        dataset_name = args.options[0]
        residue_index = int(args.options[1])

        show_dataset(dataset_name, residue_index, dont_show_plot=False)
