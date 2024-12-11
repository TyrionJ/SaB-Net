import os
import argparse

from scripts.predictor import Predictor
code_directory = os.path.dirname(__file__)


def run_predict():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', type=str, default=f'{code_directory}/../../data/SaB_raw/Dataset001_GTS/imagesTr')
    parser.add_argument('-o', type=str, default=f'{code_directory}/../../data/SaB_raw/Dataset001_GTS/preds')
    parser.add_argument('-r', type=str, default=f'{code_directory}/../../data/SaB_results', help='results folder')
    parser.add_argument('-d', type=str, default='1', help='cpu or CUDA device')
    parser.add_argument('-D', type=int, default=1, help='dataset ID')
    args = parser.parse_args()

    Predictor(dataset_id=args.D, results_folder=args.r,
              input_folder=args.i, output_folder=args.o, device=args.d).run()


if __name__ == '__main__':
    run_predict()
