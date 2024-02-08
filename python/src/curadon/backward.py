from typing import Union

import torch
import numpy as np

from . import backend as _C
from .geometry import FanGeometry, ConeGeometry


def backward(sinogram: torch.cuda.FloatTensor, geom: Union[FanGeometry, ConeGeometry], volume: torch.cuda.FloatTensor = None):
    if not torch.is_floating_point(sinogram) or not sinogram.is_cuda:
        raise TypeError(
            "Input sinogram must be a float tensor stored in CUDA memory")

    if np.any(sinogram.shape != geom.sinogram_shape()):
        raise ValueError(
            f"Input sinogram shape does not fit geometry: got {sinogram.shape}, expected {geom.sinogram_shape()}")

    # TODO: remove this?
    if volume is None:
        volume = torch.zeros(
            *geom.vol_shape, dtype=torch.float32, device=sinogram.device)

    if not torch.is_floating_point(volume) or not volume.is_cuda:
        raise TypeError(
            "Input volume must be a float tensor stored in CUDA memory")
    if np.any(volume.shape != geom.vol_shape):
        raise ValueError(
            f"Input volume shape does not fit geometry: got {volume.shape}, expected {geom.sinogram_shape()}")

    if isinstance(geom, FanGeometry):
        _C.backward_2d(volume, sinogram, geom.plan)
        return volume
    if isinstance(geom, ConeGeometry):
        _C.backward_3d(volume,
                       geom.vol_shape,
                       geom.vol_spacing,
                       geom.vol_offset,
                       sinogram,
                       geom.angles[0],
                       geom.angles[1],
                       geom.angles[2],
                       geom.det_shape,
                       geom.det_spacing,
                       geom.det_offset,
                       geom.det_rotation,
                       geom.DSO,
                       geom.DSD,
                       geom.COR)
        return volume

    raise ValueError('Unknown geometry type')
