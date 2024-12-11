import os
import argparse

from scripts.preprocessor import Processor
code_directory = os.path.dirname(__file__)

def run_preprocess():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', type=str, default=f'{code_directory}/../../data/SaB_processed', help='processed folder')
    parser.add_argument('-r', type=str, default=f'{code_directory}/../../data/SaB_raw', help='raw data folder')
    parser.add_argument('-D', type=int, default=1, help='dataset ID')
    args = parser.parse_args()

    Processor(dataset_id=args.D, raw_folder=args.r, processed_folder=args.p).run()


if __name__ == '__main__':
    run_preprocess()
