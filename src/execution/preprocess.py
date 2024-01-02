import argparse

from scripts.preprocessor import Processor


def run_preprocess():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', type=str, default='../../data/SaB_processed')
    parser.add_argument('-r', type=str, default='../../data/SaB_raw')
    parser.add_argument('-D', type=int, default=1)
    args = parser.parse_args()

    Processor(dataset_id=args.D, raw_folder=args.r, processed_folder=args.p).run()


if __name__ == '__main__':
    run_preprocess()
