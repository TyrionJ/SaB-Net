from collections import OrderedDict
from skimage.transform import resize
from batchgenerators.augmentations.utils import resize_segmentation


def resize_seg(seg, new_shape):
    return resize_segmentation(seg, new_shape, 3, **OrderedDict())


def resize_img(img, new_shape):
    return resize(img, new_shape, 3, **{'mode': 'edge', 'anti_aliasing': False})


def compute_new_shape(old_shape, old_spacing, new_spacing):
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    new_shape = [int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)]
    return new_shape
