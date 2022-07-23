import numpy as np

from dpipe.im.shape_ops import zoom
from dpipe.im.patch import sample_box_center_uniformly
from dpipe.im.box import get_centered_box
from dpipe.im.shape_ops import crop_to_box
from dpipe.dataset.wrappers import Proxy
from dpipe.im.slices import iterate_slices
from dpipe.itertools import lmap
from dpipe.batch_iter import unpack_args

SPATIAL_DIMS = (-3, -2, -1)

class Change(Proxy):
    def _change(self, x, i):
        raise NotImplementedError

    def load_image(self, i):
        return self._change(self._shadowed.load_image(i), i)

    def load_segm(self, i):
        return np.float32(self._change(self._shadowed.load_segm(i), i) >= .5)

class Rescale3D(Change):
    def __init__(self, shadowed, new_voxel_spacing=1., order=3):
        super().__init__(shadowed)
        self.new_voxel_spacing = np.broadcast_to(new_voxel_spacing, 3).astype(float)
        self.order = order

    def _scale_factor(self, i):
        old_voxel_spacing = self._shadowed.load_spacing(i)
        scale_factor = old_voxel_spacing / self.new_voxel_spacing
        return np.nan_to_num(scale_factor, nan=1)

    def _change(self, x, i):
        return zoom(x, self._scale_factor(i), order=self.order)

    def load_spacing(self, i):
        old_spacing = self.load_orig_spacing(i)
        spacing = self.new_voxel_spacing.copy()
        spacing[np.isnan(spacing)] = old_spacing[np.isnan(spacing)]
        return spacing

    def load_orig_spacing(self, i):
        return self._shadowed.load_spacing(i)



def sample_center_uniformly(shape, patch_size, spatial_dims):
    spatial_shape = np.array(shape)[list(spatial_dims)]
    if np.all(patch_size <= spatial_shape):
        return sample_box_center_uniformly(shape=spatial_shape, box_size=patch_size)
    else:
        return spatial_shape // 2

def extract_patch(inputs, x_patch_size, y_patch_size, spatial_dims=SPATIAL_DIMS):
    x, y, center = inputs

    x_patch_size = np.array(x_patch_size)
    y_patch_size = np.array(y_patch_size)
    x_spatial_box = get_centered_box(center, x_patch_size)
    y_spatial_box = get_centered_box(center, y_patch_size)

    x_patch = crop_to_box(x, box=x_spatial_box, padding_values=np.min, axis=spatial_dims)
    y_patch = crop_to_box(y, box=y_spatial_box, padding_values=0, axis=spatial_dims)
    return x_patch, y_patch

def get_random_patch_2d(image_slc, segm_slc, x_patch_size, y_patch_size):
    sp_dims_2d = (-2, -1)
    center = sample_center_uniformly(segm_slc.shape, y_patch_size, sp_dims_2d)
    x, y = extract_patch((image_slc, segm_slc, center), x_patch_size, y_patch_size, spatial_dims=sp_dims_2d)
    return x, y

def scale_mri(image: np.ndarray, q_min: int = 1, q_max: int = 99) -> np.ndarray:
    image = np.clip(np.float32(image), *np.percentile(np.float32(image), [q_min, q_max]))
    image -= np.min(image)
    image /= np.max(image)
    return np.float32(image)

def slicewise(predict):
    def wrapper(*arrays):
        return np.stack(lmap(unpack_args(predict), iterate_slices(*arrays, axis=-1)), -1)

    return wrapper

def get_random_slice(*arrays, interval: int = 1,msm=False):
    if msm:
        slc = np.random.randint(arrays[0].shape[0] // interval) * interval
        return tuple(array[slc] for array in arrays)
    else:
        slc = np.random.randint(arrays[0].shape[-1] // interval) * interval
        return tuple(array[..., slc] for array in arrays)