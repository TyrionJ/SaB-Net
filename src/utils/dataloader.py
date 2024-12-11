import os
import numpy as np
from os.path import join
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.utilities.file_and_folder_operations import load_pickle

rndst = np.random.RandomState(1234)


class NetDataloader(DataLoader):
    def __init__(self, data_folder, selected, batch_size, patch_size, in_chns=1):
        super().__init__(None, batch_size, 1, None, True, False, True)

        self.foreground_percent = 0.333
        self.data_folder = data_folder
        self.indices = self.collect_indices(selected)
        self.patch_size = patch_size
        self.seg_shape = [batch_size, ] + [in_chns, ] + patch_size
        self.lbl_shape = [batch_size, ] + [1, ] + patch_size

    def must_foreground(self, sample_idx):
        return not sample_idx < round(self.batch_size * (1 - self.foreground_percent))

    def collect_indices(self, selected):
        data_keys = [f[:-4] for f in os.listdir(self.data_folder) if f.endswith('.pkl')]
        return sorted([k for k in data_keys if k in selected])

    def generate_train_batch(self):
        selected_keys = self.get_indices()

        data_all = np.zeros(self.seg_shape, dtype=np.float32)
        label_all = np.zeros(self.lbl_shape, dtype=np.int16)

        for i, key in enumerate(selected_keys):
            data = np.load(join(self.data_folder, f'{key}_img.npy')).astype(float)
            label = np.load(join(self.data_folder, f'{key}_seg.npy')).astype(float)
            pkl = load_pickle(join(self.data_folder, f'{key}.pkl'))
            must_fg = self.must_foreground(i)

            shape = data.shape[1:]
            bbox_lbs, bbox_ubs = get_bbox(shape, must_fg, pkl['class_locs'], self.patch_size)

            data_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(bbox_lbs, bbox_ubs)])
            data = data[data_slice]

            lbl_slice = tuple([slice(0, label.shape[0])] + [slice(i, j) for i, j in zip(bbox_lbs, bbox_ubs)])
            label = label[lbl_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(3)]
            data_all[i] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            label_all[i] = np.pad(label, ((0, 0), *padding), 'constant', constant_values=0)

        return {'seg_data': data_all, 'seg_label': label_all}


def get_bbox(shape, must_fg, class_pos, patch_size):
    lbs, ubs = [0, 0, 0], [max(i-j, 0) for i, j in zip(shape, patch_size)]
    if not must_fg:
        bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(3)]
    else:
        eligible_classes_or_regions = [i for i in class_pos.keys() if len(class_pos[i]) > 0]
        selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))]
        voxels_of_that_class = class_pos[selected_class]
        selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
        bbox_lbs = [max(lbs[i], selected_voxel[i] - patch_size[i] // 2) for i in range(3)]
    bbox_ubs = [bbox_lbs[i] + patch_size[i] for i in range(3)]

    return bbox_lbs, bbox_ubs
