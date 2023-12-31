import os.path
from batchgenerators.utilities.file_and_folder_operations import save_json, load_json, save_pickle, load_pickle


__all__ = ['save_pickle', 'save_json', 'load_json', 'load_pickle', 'maybe_mkdir']


def maybe_mkdir(Dir):
    if not os.path.exists(Dir):
        os.makedirs(Dir)
