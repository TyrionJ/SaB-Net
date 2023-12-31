import os
import numpy as np
import SimpleITK as sitk
import multiprocessing as mp
from os.path import join, isdir, exists
from sklearn.model_selection import KFold

from utils.normalization import CTNormalize
from utils.waiting_process import waiting_proc
from utils.resampling import resize_seg, resize_img, compute_new_shape
from utils.folder_file_operator import maybe_mkdir, load_json, save_json, save_pickle
from utils.wellcome import wellcome

num_foreground_voxels = 10e7
rs = np.random.RandomState(123456)


class Processor:
    def __init__(self, dataset_id, raw_folder, processed_folder, logger=None):
        self.dataset_name = self.get_dataset_name(dataset_id, raw_folder)
        self.raw_dataset = join(raw_folder, self.dataset_name)
        self.processed_dataset = join(processed_folder, self.dataset_name)
        maybe_mkdir(self.processed_dataset)
        self.logger = logger or print
        wellcome()

    @staticmethod
    def get_dataset_name(dataset_id, raw_folder):
        assert isdir(raw_folder), "The requested raw data folder could not be found"
        for dataset in os.listdir(raw_folder):
            if f'{dataset_id:03d}' in dataset:
                return dataset
        raise f'The requested dataset {dataset_id} could not be found in sab_raw'

    @staticmethod
    def analysis_case(image_file, seg_file, num_samples):
        itk_img = sitk.ReadImage(image_file)
        itk_seg = sitk.ReadImage(seg_file)

        npy_img = sitk.GetArrayFromImage(itk_img)
        npy_seg = sitk.GetArrayFromImage(itk_seg)
        foreground_pixels = rs.choice(npy_img[npy_seg == 1], num_samples, replace=True)

        return itk_img.GetSpacing()[::-1], foreground_pixels

    @staticmethod
    def sample_foreground_locations(seg: np.ndarray, classes):
        num_samples = 10000
        min_percent_coverage = 0.01

        class_locs = {}
        for c in classes:
            k = c if not isinstance(c, list) else tuple(c)
            if isinstance(c, (tuple, list)):
                mask = seg == c[0]
                for cc in c[1:]:
                    mask = mask | (seg == cc)
                all_locs = np.argwhere(mask)
            else:
                all_locs = np.argwhere(seg == c)
            if len(all_locs) == 0:
                class_locs[k] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

            selected = all_locs[rs.choice(len(all_locs), target_num_samples, replace=False)]
            class_locs[k] = selected
        return class_locs

    def extract_dactylogram(self):
        self.logger(' Processing dactylogram ...')
        dtg_file = join(self.processed_dataset, 'dactylogram.json')
        if exists(dtg_file):
            return load_json(dtg_file)

        imgs_folder = join(self.raw_dataset, 'imagesTr')
        lbls_folder = join(self.raw_dataset, 'labelsTr')
        img_keys = [i[:-7] for i in sorted(os.listdir(imgs_folder)) if i.endswith('.nii.gz')]
        voxels_per_case = int(num_foreground_voxels // len(img_keys))

        r = []
        with mp.get_context('spawn').Pool(12) as p:
            for img_key in img_keys:
                image_file = join(imgs_folder, f'{img_key}.nii.gz')
                seg_file = join(lbls_folder, f'{img_key}.nii.gz')
                r.append(p.starmap_async(self.analysis_case, ((image_file, seg_file, voxels_per_case),)))
            waiting_proc(r, p)

        results = [i.get()[0] for i in r]
        spacings = [i[0] for i in results]
        foregrounds = np.concatenate([i[1] for i in results])

        dtg_obj = {
            'foreground_properties': {
                'mean': float(np.mean(foregrounds)),
                'std': float(np.std(foregrounds)),
                'percentile_99_5': float(np.percentile(foregrounds, 99.5)),
                'percentile_00_5': float(np.percentile(foregrounds, 0.5)),
            },
            'spacing': list(np.median(spacings, axis=0))
        }
        save_json(dtg_obj, dtg_file)

        return dtg_obj

    def save_data(self, image_file, seg_file, foreground_properties, new_spacing, img_key):
        itk_img = sitk.ReadImage(image_file)
        itk_seg = sitk.ReadImage(seg_file)

        npy_img = sitk.GetArrayFromImage(itk_img)
        npy_seg = sitk.GetArrayFromImage(itk_seg)
        spacing = itk_img.GetSpacing()[::-1]
        ori_shape = npy_img.shape
        new_shape = compute_new_shape(ori_shape, spacing, new_spacing)
        npy_img = CTNormalize(npy_img, foreground_properties)

        npy_img = resize_img(npy_img, new_shape).astype(np.float32)
        npy_seg = resize_seg(npy_seg, new_shape).astype(np.int8)

        to_fdr = join(self.processed_dataset, 'data')
        np.save(join(to_fdr, f'{img_key}_img.npy'), npy_img[None])
        np.save(join(to_fdr, f'{img_key}_seg.npy'), npy_seg[None])
        save_pickle({
            'original_shape': ori_shape, 'spacing': spacing,
            'class_locs': self.sample_foreground_locations(npy_seg, [1])
        }, join(to_fdr, f'{img_key}.pkl'))

    def split_dataset(self, img_keys):
        self.logger(' Splitting dataset ...')
        if not exists(join(self.processed_dataset, 'splits.json')):
            splits = []
            kfold = KFold(n_splits=5, shuffle=True, random_state=20184)
            for i, (train_idx, test_idx) in enumerate(kfold.split(img_keys)):
                train_keys = np.array(img_keys)[train_idx]
                test_keys = np.array(img_keys)[test_idx]
                splits.append({
                    'train': list(train_keys),
                    'val': list(test_keys)
                })
            save_json(splits, join(self.processed_dataset, 'splits.json'))

    def process_data(self, fpts, img_keys):
        self.logger(' Processing data ...')
        imgs_folder = join(self.raw_dataset, 'imagesTr')
        lbls_folder = join(self.raw_dataset, 'labelsTr')
        maybe_mkdir(join(self.processed_dataset, 'data'))
        r = []
        with mp.get_context('spawn').Pool(12) as p:
            for img_key in img_keys:
                image_file = join(imgs_folder, f'{img_key}.nii.gz')
                seg_file = join(lbls_folder, f'{img_key}.nii.gz')
                fp = fpts['foreground_properties']
                spacing = fpts['spacing']
                r.append(p.starmap_async(self.save_data, ((image_file, seg_file, fp, spacing, img_key),)))
            waiting_proc(r, p)

    def run(self):
        self.logger(f'Preprocessing dataset {self.dataset_name} ...')
        img_keys = [i[:-7] for i in sorted(os.listdir(join(self.raw_dataset, 'imagesTr'))) if i.endswith('.nii.gz')]

        dtg = self.extract_dactylogram()
        self.process_data(dtg, img_keys)
        self.split_dataset(img_keys)
