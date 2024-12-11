import os
import torch
import numpy as np
from typing import List


class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def empty_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    else:
        pass


def say_hi(logger):
    logger("\n###################################################################################################\n"
           " Please cite the following paper when using SaB-Net:\n"
           " He, J., Zhang, M., Li, W., Peng, Y., Fu, B., Liu, C., Wang, J., & Wang, R.\n"
           " SaB-Net: Self-attention backward network for gastric tumor segmentation in CT images[J].\n"
           " Computers in Biology and Medicine, 2023: 107866. https://doi.org/10.1016/j.compbiomed.2023.107866\n"
           "###################################################################################################\n\n")


def collate_outputs(outputs: List[dict]):
    collated = {}
    for k in outputs[0].keys():
        if np.isscalar(outputs[0][k]):
            collated[k] = [o[k] for o in outputs]
        elif isinstance(outputs[0][k], np.ndarray):
            collated[k] = np.vstack([o[k][None] for o in outputs])
        elif isinstance(outputs[0][k], list):
            collated[k] = [item for o in outputs for item in o[k]]
        else:
            raise ValueError(f'Cannot collate input of type {type(outputs[0][k])}. '
                             f'Modify collate_outputs to add this functionality')
    return collated


def get_allowed_n_proc():
    return min(16, os.cpu_count())
