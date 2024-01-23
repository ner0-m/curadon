from typing import Union

import torch
import numpy as np

from . import backend as _C
from .geometry import FanGeometry, ConeGeometry


def forward(volume: torch.cuda.FloatTensor, geom: Union[FanGeometry, ConeGeometry], sinogram: torch.cuda.FloatTensor = None):
    if not isinstance(volume, torch.cuda.FloatTensor) or volume.is_cuda is False or volume.dtype != torch.float32:
        raise TypeError(
            "Input volume must be a float tensor stored in CUDA memory")
    if np.any(volume.shape != geom.vol_shape):
        raise ValueError(
            f"Input volume shape does not fit geometry: got {volume.shape} expected {geom.vol_shape}")

    if sinogram is None:
        sinogram = torch.zeros(*geom.sinogram_shape(),
                               dtype=torch.float32, device=volume.device)
    if not isinstance(volume, torch.cuda.FloatTensor) or volume.is_cuda is False or sinogram.dtype != torch.float32:
        raise TypeError(
            "Input sinogram must be a float tensor stored in CUDA memory")
    if sinogram.shape != geom.sinogram_shape():
        raise ValueError(
            f"Input sinogram shape does not fit geometry: got {sinogram.shape}, expected {geom.sinogram_shape()}")

    if isinstance(geom, FanGeometry):
        _C.forward_2d(volume,
                      geom.vol_shape,
                      geom.vol_spacing,
                      geom.vol_offset,
                      sinogram,
                      geom.angles.cpu(),
                      geom.det_shape,
                      geom.det_spacing,
                      geom.det_offset,
                      geom.det_rotation,
                      geom.DSO,
                      geom.DSD,
                      geom.COR)
        return sinogram
    if isinstance(geom, ConeGeometry):
        _C.forward_3d(volume,
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
        return sinogram
    raise ValueError("Unknown geometry type")
