import torch
import os.path
from batchgenerators.utilities.file_and_folder_operations import save_json, load_json, save_pickle, load_pickle


__all__ = ['save_pickle', 'save_json', 'load_json', 'load_pickle', 'maybe_mkdir', 'empty_cache', 'dummy_context']


def maybe_mkdir(Dir):
    if Dir and not os.path.exists(Dir):
        os.makedirs(Dir)


def empty_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        from torch import mps
        mps.empty_cache()
    else:
        pass


class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
