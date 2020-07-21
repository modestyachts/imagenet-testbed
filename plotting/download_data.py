import argparse
import os
import sys
import pathlib
from timeit import default_timer as timer

from dataframe import download_grid_dataframe


def download_plotting_data(output_dir=None, store_data=False, verbose=False):
    if verbose:
        print('Downloading the grid data from the database ... ', flush=True)

    start = timer()
    df = download_grid_dataframe()
    end = timer()
    
    if verbose:
        print(f'done, took {end - start:.2f} seconds')
    
    if store_data:
        assert output_dir is not None
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            if verbose:
                print(f'Created the directory {output_dir}')

        dest = os.path.join(output_dir, 'grid_df.pkl')
        df.to_pickle(dest)
        if verbose:
            print(f'Wrote a dataframe with {df.shape[0]} rows and {df.shape[1]} columns to {dest}')

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default=str((pathlib.Path(__file__).parent / '../outputs').resolve()))
    args = parser.parse_args()

    download_plotting_data(args.logdir, store_data=True, verbose=True)
