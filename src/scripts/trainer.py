import time
import torch
import os.path
import warnings
import numpy as np
from tqdm import tqdm
from typing import List
from datetime import datetime
from os.path import join, isdir
from torch.cuda.amp import GradScaler
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json, maybe_mkdir_p, load_pickle

from network import SaBNet
from net_loss import NetLoss
from scripts.predictor import Predictor
from utils.dataloader import NetDataloader
from utils.evaluation import get_tp_fp_fn_tn
from utils.polyrescheduler import PolyLRScheduler
from transforms.train_transform import TrTransform, VdTransform
from transforms.limited_length_multithreaded_augmenter import LimitedLenWrapper
from utils.helpers import empty_cache, say_hi, collate_outputs, get_allowed_n_proc

warnings.filterwarnings('ignore')


class NetTrainer:
    network = optimizer = lr_scheduler = None
    processed_folder = result_dataset = fold_fdr = final_valid_fdr = None

    def __init__(self, batch_size, patch_size, processed_folder, dataset_id,
                 result_folder, fold, go_on, epochs, device, validation=False, logger=print):
        
        self.in_chs = 1
        self.out_chs = 2
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.fold = fold
        self.go_on = go_on
        self.epochs = epochs
        self.validation = validation
        self.dataset_id = dataset_id
        self.result_folder = result_folder
        self.device = torch.device(f'cuda:{device}') if device != 'cpu' else torch.device(device)

        self.install_folder(processed_folder)
        self.save_model_info()
        self.logger = self.build_logger(logger)
        say_hi(self.logger)

        self.ds_scales = [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25],
                          [0.125, 0.125, 0.125], [0.0625, 0.0625, 0.0625]]
        self.cur_epoch = 0
        self.initial_lr = 1e-2      # 1e-4
        self.weight_decay = 3e-5    # 1e-5
        self.train_iters = 250
        self.valid_iters = 50
        self.save_interval = 1
        self.best_dice = 0

        self.train_loader, self.valid_loader = self.get_tr_vd_loader()
        self.grad_scaler = GradScaler() if device != 'cpu' else None
        self.loss_fn = NetLoss(self.ds_scales)

    def build_logger(self, logger):
        now = datetime.now()
        prefix = 'training' if not self.validation else 'validation'
        log_file = join(self.fold_fdr, f'{prefix}_log_{now.strftime("%Y-%m-%d_%H-%M-%S")}.txt')
        fw = open(log_file, 'a', encoding='utf-8')

        def log_fn(content):
            logger(content)
            fw.write(f'{content}\n')
            fw.flush()

        return log_fn

    def install_folder(self, processed_folder):
        assert 0 <= self.fold < 5, 'only support 5-fold training, and fold should belong to [0, 5)'
        assert isdir(processed_folder), "The requested processed data folder could not be found"

        d_name = None
        for dataset in os.listdir(processed_folder):
            if f'{self.dataset_id:03d}' in dataset:
                d_name = dataset
                break
        assert d_name is not None, f'The requested dataset {self.dataset_id} could not be found in processed_folder'

        self.processed_folder = join(processed_folder, d_name)
        self.result_dataset: str = join(self.result_folder, d_name)
        self.fold_fdr = join(self.result_dataset, f'fold_{self.fold}')
        self.final_valid_fdr = join(self.fold_fdr, 'validation')
        maybe_mkdir_p(self.final_valid_fdr)

    def save_model_info(self):
        info = {
            'in_chs': self.in_chs,
            'out_chs':  self.out_chs,
            'patch_size': self.patch_size,
            'dactylogram': load_json(join(self.processed_folder, 'dactylogram.json'))
        }
        save_json(info, join(self.result_dataset, 'model_info.json'))

    def get_tr_vd_indices(self, verbose=True):
        s_file = join(self.processed_folder, 'splits.json')
        splits = load_json(s_file)
        fold = splits[self.fold]

        if verbose:
            self.logger(f'Use splits: {s_file}')
            self.logger(f'The file contains {len(splits)} splits.')
            self.logger(f'Fold for training: {self.fold}')
        return fold['train'], fold['val']

    def get_tr_vd_loader(self):
        train_indices, valid_indices = self.get_tr_vd_indices()
        self.logger(f"tr_set size={len(train_indices)}, val_set size={len(valid_indices)}")
        data_fdr = join(self.processed_folder, 'data')
        tr_loader = NetDataloader(data_fdr, train_indices, self.batch_size, self.patch_size, self.in_chs)
        vd_loader = NetDataloader(data_fdr, valid_indices, max(2, self.batch_size // 2), self.patch_size, self.in_chs)
        tr_transforms, val_transforms = TrTransform(self.ds_scales), VdTransform(self.ds_scales)

        allowed_num_processes = get_allowed_n_proc()
        if allowed_num_processes == 0 or self.device.type == 'cpu':
            mt_gen_train = SingleThreadedAugmenter(tr_loader, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(vd_loader, val_transforms)
        else:
            mt_gen_train = LimitedLenWrapper(self.train_iters, data_loader=tr_loader,
                                             transform=tr_transforms,
                                             num_processes=allowed_num_processes, num_cached=6, seeds=None,
                                             pin_memory=self.device.type == 'cuda')
            mt_gen_val = LimitedLenWrapper(self.valid_iters, data_loader=vd_loader,
                                           transform=val_transforms, num_processes=max(1, allowed_num_processes // 2),
                                           num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda')
            time.sleep(0.1)
        return mt_gen_train, mt_gen_val

    def initialize(self):
        empty_cache(self.device)
        self.network = SaBNet(self.in_chs, self.out_chs).to(self.device)
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = PolyLRScheduler(self.optimizer, self.initial_lr, self.epochs)
        self.load_states()

    def load_states(self):
        check_file = join(self.fold_fdr, 'model_latest.pt' if not self.validation else 'model_best.pt')
        if (self.go_on or self.validation) and os.path.isfile(check_file):
            self.logger(f'Use checkpoint: {check_file}')
            weights = torch.load(join(self.fold_fdr, 'model_latest.pt'), map_location=torch.device('cpu'))
            checkpoint = torch.load(join(self.fold_fdr, 'check_latest.pth'), map_location=torch.device('cpu'))

            if 'cur_epoch' in weights:
                del weights['cur_epoch']
            self.network.load_state_dict(weights)
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.cur_epoch = checkpoint['cur_epoch']
            self.best_dice = checkpoint['best_dice']
            if self.grad_scaler is not None and checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

    def save_states(self, val_dice):
        self.cur_epoch += 1
        checkpoint = {
            'optimizer_state': self.optimizer.state_dict(),
            'cur_epoch': self.cur_epoch,
            'best_dice': self.best_dice,
            'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None
        }
        if self.grad_scaler is not None:
            checkpoint['grad_scaler_state'] = self.grad_scaler.state_dict()

        if self.best_dice < val_dice:
            self.best_dice = val_dice
            self.logger(f'Eureka!!! Best dice: {self.best_dice:.4f}')
            torch.save(self.network.state_dict(), join(self.fold_fdr, 'model_best.pt'))

        if self.cur_epoch % self.save_interval == 0 or self.cur_epoch == self.epochs:
            torch.save(checkpoint, join(self.fold_fdr, 'check_latest.pth'))
            torch.save(self.network.state_dict(), join(self.fold_fdr, 'model_latest.pt'))
        self.logger('')

    def train_step(self, batch: dict) -> dict:
        seg_data = batch['seg_data']
        tgt_data = batch['seg_label']

        seg_data = seg_data.to(self.device)
        if isinstance(tgt_data, list):
            tgt_data = [i.to(self.device, ) for i in tgt_data]
        else:
            tgt_data = tgt_data.to(self.device)

        self.optimizer.zero_grad()

        net_out = self.network(seg_data, self.cur_epoch)
        t_loss = self.loss_fn(net_out, tgt_data)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(t_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            t_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': t_loss.detach().cpu().numpy()}

    def valid_step(self, batch: dict) -> dict:
        seg_data = batch['seg_data']
        tgt_data = batch['seg_label']

        seg_data = seg_data.to(self.device)
        if isinstance(tgt_data, list):
            tgt_data = [i.to(self.device) for i in tgt_data]
        else:
            tgt_data = tgt_data.to(self.device)

        net_out = self.network(seg_data)
        t_loss = self.loss_fn(net_out, tgt_data)

        seg_out = net_out[0]
        target = tgt_data[0]

        axes = [0] + list(range(2, len(seg_out.shape)))
        output_seg = seg_out.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(seg_out.shape, device=seg_out.device, dtype=torch.float32)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        del output_seg

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes)

        tp_hard = tp.detach().cpu().numpy()[1:]
        fp_hard = fp.detach().cpu().numpy()[1:]
        fn_hard = fn.detach().cpu().numpy()[1:]

        return {'loss': t_loss.detach().cpu().numpy(),
                'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def on_train_epoch_end(self, epoch, train_loss, lr):
        self.logger(f'Epoch: {epoch} / {self.epochs}')
        self.logger(f'current lr: {np.round(lr, decimals=5)}')
        self.logger(f'train loss: {np.round(train_loss, decimals=6)}')

    def on_valid_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        loss_here = np.mean(outputs_collated['loss']).astype(np.float64)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k + 1e-8) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)

        self.logger(f'validation loss: {np.round(loss_here, decimals=6)}')
        self.logger(f'valid mean Dice: {np.round(mean_fg_dice, decimals=6)}')
        self.logger(f'vDice per class: {[np.round(i, decimals=2) for i in global_dc_per_class]}')

        return mean_fg_dice

    def on_train_end(self):
        final = {
            'network_weights': torch.load(join(self.fold_fdr, 'model_best.pt')),
            'dactylogram': load_json(join(self.processed_folder, 'dactylogram.json'))
        }
        torch.save(final, join(self.fold_fdr, 'checkpoint_final.pth'))

    def conduct_final_validation(self):
        self.logger('\nFinal Validation:')

        predictor = Predictor(self.dataset_id, self.result_folder, '', '', self.device.index or 'cpu')
        predictor.network = self.network

        train_indices, valid_indices = self.get_tr_vd_indices(False)
        mean_Dice = 0
        for N, key in enumerate(valid_indices):
            self.logger(f'Validating {key}:')
            seg_data = np.load(join(self.processed_folder, 'data', f'{key}_img.npy'))
            lbl_data = np.load(join(self.processed_folder, 'data', f'{key}_seg.npy'))
            pkl_info = load_pickle(join(self.processed_folder, 'data', f'{key}.pkl'))

            in_data = torch.from_numpy(seg_data).float()
            y_logics = predictor.sliding_return_logits(in_data, '  State')
            y_pred = torch.softmax(y_logics.float(), 0).argmax(0).cpu().numpy().astype(float)

            segm_onehot = torch.zeros((1, self.out_chs) + y_pred.shape, dtype=torch.float32)
            segm_onehot.scatter_(1, torch.from_numpy(y_pred[None, None].astype(np.int64)), 1)
            tp, fp, fn, _ = get_tp_fp_fn_tn(segm_onehot, torch.from_numpy(lbl_data[None]), axes=(0, 2, 3, 4))
            tp = tp.numpy()[1:]
            fp = fp.numpy()[1:]
            fn = fn.numpy()[1:]
            DCpC = [i for i in [2 * i / (2 * i + j + k + 1e-8) for i, j, k in zip(tp, fp, fn)]]
            m_DC = np.mean(DCpC)

            mean_Dice = (mean_Dice * N + m_DC) / (N + 1)
            self.logger(f'  AvDC={np.round(m_DC, decimals=6)}')
            self.logger(f'  DCpC={[np.round(i, decimals=2) for i in DCpC]}')

            to_file = join(self.final_valid_fdr, f'{key}_vald.nii.gz')
            meta_info = (pkl_info['original_shape'], pkl_info['ori_spacing'], pkl_info['origin'], pkl_info['direction'])
            predictor.save_logits_to_segmentation(y_logics, meta_info, to_file)
            self.logger('')

        self.logger('Final Validation complete')
        self.logger(f'  Mean Validation Dice: {np.round(mean_Dice, decimals=6)}')

    def run(self):
        self.initialize()

        if not self.validation:
            self.logger('\nBegin training ...')
            time.sleep(0.5)

            self.network.deep_supervision = True
            for epoch in range(self.cur_epoch, self.epochs):
                avg_loss = 0

                self.lr_scheduler.step(self.cur_epoch)
                lr = self.optimizer.param_groups[0]['lr']

                self.network.train()
                with tqdm(desc=f'[{epoch + 1}/{self.epochs}]Training', total=self.train_iters) as p:
                    for batch_id in range(self.train_iters):
                        train_loss = self.train_step(next(self.train_loader))['loss']
                        avg_loss = (avg_loss * batch_id + train_loss) / (batch_id + 1)
                        p.set_postfix(**{'avg': '%.4f' % avg_loss, 'bat': '%.4f' % train_loss, 'lr': '%.6f' % lr})
                        p.update()

                self.network.eval()
                with torch.no_grad():
                    with tqdm(desc='~~Validation', total=self.valid_iters, colour='green') as p:
                        val_outputs = []
                        for batch_id in range(self.valid_iters):
                            val_outputs.append(self.valid_step(next(self.valid_loader)))
                            p.update()

                self.on_train_epoch_end(epoch+1, avg_loss, lr)
                val_dice = self.on_valid_epoch_end(val_outputs)
                self.save_states(val_dice)
            self.on_train_end()
            self.logger('Training end!')

        self.conduct_final_validation()
