import torch.nn as nn

from .seg_loss import SegLoss


class NetLoss(nn.Module):
    def __init__(self, ds_scales):
        super().__init__()
        self.seg_loss = SegLoss(ds_scales)

    def forward(self, net_out, target):
        s_loss = self.seg_loss([net_out, target])

        return s_loss
