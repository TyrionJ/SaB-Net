import argparse
from scripts.predictor import Predictor


def run_predict():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', type=str, default='../../data/SaB_raw/Dataset001_GTS/imagesTr')
    parser.add_argument('-o', type=str, default='../../data/SaB_raw/Dataset001_GTS/preds')
    parser.add_argument('-r', type=str, default='../../data/SaB_results')
    parser.add_argument('-d', type=str, default='0')
    parser.add_argument('-D', type=int, default=1)
    args = parser.parse_args()

    Predictor(dataset_id=args.D, results_folder=args.r,
              input_folder=args.i, output_folder=args.o, device=args.d).run()


if __name__ == '__main__':
    run_predict()
