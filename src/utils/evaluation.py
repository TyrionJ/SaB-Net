import torch
import numpy as np


def get_tp_fp_fn_tn(net_output, gt, axes=None):
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x, device=net_output.device)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keep_dim=False)
        fp = sum_tensor(fp, axes, keep_dim=False)
        fn = sum_tensor(fn, axes, keep_dim=False)
        tn = sum_tensor(tn, axes, keep_dim=False)

    return tp, fp, fn, tn


def sum_tensor(inp: torch.Tensor, axes, keep_dim: bool = False) -> torch.Tensor:
    axes = np.unique(axes).astype(int)
    if keep_dim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp
