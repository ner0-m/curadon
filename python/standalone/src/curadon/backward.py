from typing import Union

from . import backend as _C
from .geometry import FanGeometry, ConeGeometry


def backward(sinogram, volume, geom: Union[FanGeometry, ConeGeometry]):
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
