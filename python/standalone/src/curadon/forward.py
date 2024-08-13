from typing import Union

from . import backend as _C
from .geometry import FanGeometry, ConeGeometry


def forward(volume, sinogram, geom: Union[FanGeometry, ConeGeometry]):
    if isinstance(geom, FanGeometry):
        _C.forward_2d(volume, sinogram, geom.plan)
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
