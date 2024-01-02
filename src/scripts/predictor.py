import os
import torch
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from functools import lru_cache
from skimage.transform import resize
from os.path import join, isdir, isfile
from scipy.ndimage import gaussian_filter
from acvl_utils.cropping_and_padding.padding import pad_nd_image

from network import SaBNet
from .preprocessor import Processor
from utils.wellcome import wellcome
from utils.folder_file_operator import maybe_mkdir, empty_cache, dummy_context


class Predictor:
    def __init__(self, dataset_id, results_folder, input_folder, output_folder, device, logger=None):
        self.dataset_name = self.get_dataset_name(dataset_id, results_folder)
        self.result_dataset = join(results_folder, self.dataset_name)
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.logger = logger or print
        self.device = torch.device('cpu') if device == 'cpu' else torch.device(f'cuda:{device}')

        self.dtg = None
        self.in_chs = 1
        self.out_chs = 2
        self.use_gaussian = True
        self.use_mirroring = True
        self.tile_step_size = 0.5
        self.patch_size = (64, 256, 256)
        self.network = SaBNet(in_chs=self.in_chs, out_chs=self.out_chs)

        self.load_dactylogram()
        wellcome(self.logger)
        maybe_mkdir(output_folder)

    @staticmethod
    def get_dataset_name(dataset_id, results_folder):
        assert isdir(results_folder), "The requested results folder could not be found"
        for dataset in os.listdir(results_folder):
            if f'{dataset_id:03d}' in dataset:
                return dataset
        raise f'The requested dataset {dataset_id} could not be found in {results_folder}'

    def processed_generator(self):
        image_list = sorted([i for i in os.listdir(self.input_folder) if i.endswith('.nii.gz')])

        self.logger(f'There are {len(image_list)} case(s) to predict:\n')

        for img_file in image_list:
            img_itk = sitk.ReadImage(join(self.input_folder, img_file))
            img_data = sitk.GetArrayFromImage(img_itk)
            spacing = img_itk.GetSpacing()
            origin = img_itk.GetOrigin()
            direction = img_itk.GetDirection()
            pro_data, _ = Processor.run_case(img_data, None, self.dtg, spacing[::-1])

            yield torch.from_numpy(pro_data[None]).float(), (img_data.shape, spacing, origin, direction), img_file

    def load_dactylogram(self):
        folds = [i for i in os.listdir(self.result_dataset) if i.startswith('fold_')
                 and isfile(join(self.result_dataset, i, 'checkpoint_final.pth'))]
        assert len(folds) > 0, 'No valid checkpoint'
        checkpoint = torch.load(join(self.result_dataset, folds[0], 'checkpoint_final.pth'), map_location='cpu')
        self.dtg = checkpoint['dactylogram']

    def sliding_return_logits(self, input_image, fold):
        self.network = self.network.to(self.device)
        self.network.eval()
        empty_cache(self.device)

        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                assert len(input_image.shape) == 4, 'input image must be a 4D tensor'
                data, revert_padding = pad_nd_image(input_image, self.patch_size, 'constant', {'value': 0}, True, None)
                slicers = self._get_sliding_window_slicers(data.shape[1:])

                data = data.to(self.device)
                predicted_logits = torch.zeros((self.out_chs, *data.shape[1:]), dtype=torch.half, device=self.device)
                n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=self.device)
                if self.use_gaussian:
                    gaussian = compute_gaussian(self.patch_size, sigma_scale=1./8, value_scaling_factor=1000,
                                                device=self.device)

                empty_cache(self.device)
                for sl in tqdm(slicers, desc=f'  {fold}', colour='green'):
                    workon = data[sl][None].to(self.device, non_blocking=False)
                    prediction = self._mirror_and_predict(workon)[0]

                    predicted_logits[sl] += (prediction * gaussian if self.use_gaussian else prediction)
                    n_predictions[sl[1:]] += (gaussian if self.use_gaussian else 1)
                predicted_logits /= n_predictions

        empty_cache(self.device)
        return predicted_logits[tuple([slice(None), *revert_padding[1:]])]

    def logits_from_preprocessed(self, data):
        folds = [i for i in os.listdir(self.result_dataset) if i.startswith('fold_')
                 and isfile(join(self.result_dataset, i, 'checkpoint_final.pth'))]
        with torch.no_grad():
            prediction = None
            for fold in sorted(folds):
                checkpoint = torch.load(join(self.result_dataset, fold, 'checkpoint_final.pth'), map_location='cpu')
                self.network.load_state_dict(checkpoint['network_weights'])

                if prediction is None:
                    prediction = self.sliding_return_logits(data, fold)
                else:
                    prediction += self.sliding_return_logits(data, fold)

        return prediction / len(folds)

    def run(self):
        processed_generator = self.processed_generator()

        for data, meta_info, img_file in processed_generator:
            self.logger(f'Predicting {img_file[:-7]}:')
            self.logger(f'  Data shape: {list(meta_info[0])}, Input shape: {list(data.shape)}')

            logits = self.logits_from_preprocessed(data)
            to_file = join(self.output_folder, img_file.replace('_0000', ''))
            self.save_logits_to_segmentation(logits, meta_info, to_file)
            self.logger(f'done with {img_file[:-7]}\n')

            compute_gaussian.cache_clear()
            empty_cache(self.device)

    def _mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        prediction = self.network(x)

        prediction += torch.flip(self.network(torch.flip(x, (2,))), (2,))
        prediction += torch.flip(self.network(torch.flip(x, (3,))), (3,))
        prediction += torch.flip(self.network(torch.flip(x, (4,))), (4,))
        prediction += torch.flip(self.network(torch.flip(x, (2, 3))), (2, 3))
        prediction += torch.flip(self.network(torch.flip(x, (2, 4))), (2, 4))
        prediction += torch.flip(self.network(torch.flip(x, (3, 4))), (3, 4))
        prediction += torch.flip(self.network(torch.flip(x, (2, 3, 4))), (2, 3, 4))

        return prediction / 8

    def _get_sliding_window_slicers(self, image_size):
        slicers = []
        steps = compute_steps_for_sliding_window(image_size, self.patch_size, self.tile_step_size)
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicer = tuple(
                        [slice(None), *[slice(si, si + ti) for si, ti in zip((sx, sy, sz), self.patch_size)]])
                    slicers.append(slicer)
        return slicers

    def save_logits_to_segmentation(self, logits, meta_info, to_file):
        self.logger('  Exporting result')
        shape, spacing, origin, direction = meta_info
        reshaped_logits = torch.zeros((logits.shape[0],) + shape)
        for c in range(logits.shape[0]):
            t = resize(logits[c].cpu().numpy(), shape, mode='edge', anti_aliasing=False, order=1)
            reshaped_logits[c] = torch.from_numpy(t)
        probabilities = torch.softmax(reshaped_logits, 0)
        segmentation = probabilities.argmax(0).float().numpy()

        seg_itk = sitk.GetImageFromArray(segmentation)
        seg_itk.SetSpacing(spacing)
        seg_itk.SetOrigin(origin)
        seg_itk.SetDirection(direction)
        sitk.WriteImage(seg_itk, to_file)


def compute_steps_for_sliding_window(image_size, tile_size, tile_step_size):
    assert [i >= j for i, j in zip(image_size, tile_size)], "image size must be as large or larger than patch_size"
    assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]
    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999
        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
        steps.append(steps_here)
    return steps


@lru_cache(maxsize=2)
def compute_gaussian(tile_size, sigma_scale=1. / 8,  value_scaling_factor=1, dtype=torch.float16, device=torch.device('cuda', 0)):
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1

    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = torch.from_numpy(gaussian_importance_map).type(dtype).to(device)
    gaussian_importance_map = gaussian_importance_map / torch.max(gaussian_importance_map) * value_scaling_factor
    gaussian_importance_map = gaussian_importance_map.type(dtype)
    gaussian_importance_map[gaussian_importance_map == 0] = torch.min(gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map
