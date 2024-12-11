import os
import argparse

from scripts.trainer import NetTrainer
code_directory = os.path.dirname(__file__)


def run_trainer():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', type=str, default=f'{code_directory}/../../data/SaB_processed', help='processed folder')
    parser.add_argument('-r', type=str, default=f'{code_directory}/../../data/SaB_results', help='results folder')
    parser.add_argument('-f', type=int, default=0, help='fold')
    parser.add_argument('-D', type=int, default=1, help='dataset ID')

    parser.add_argument('-b', type=int, default=2, help='batch size')
    parser.add_argument('--c', action='store_true', help='continue train')
    parser.add_argument('--v', action='store_true', help='only validation if train finished')
    parser.add_argument('-d', type=str, default='0', help='device: cpu or 0, 1, 2, ...')
    parser.add_argument('-e', type=int, default=500, help='epoch number')
    args = parser.parse_args()

    tr = NetTrainer(batch_size=args.b,
                    patch_size=[64, 256, 256],
                    processed_folder=args.p,
                    dataset_id=args.D,
                    result_folder=args.r,
                    fold=args.f,
                    go_on=args.c,
                    epochs=args.e,
                    device=args.d,
                    validation=args.v,
                    logger=print)
    tr.run()


if __name__ == '__main__':
    run_trainer()
