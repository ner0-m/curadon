import numpy as np
import torch
import astra
from phantominator import shepp_logan


def config(n):
    return {
        "vol_shape": np.array([n, n], dtype=np.uint32),
        "vol_spacing": np.array([1, 1], dtype=np.float32),
        "vol_offset": np.array([0, 0], dtype=np.float32),
        "det_shape": int(n * np.sqrt(2)),
        "det_spacing": 1,
        "det_offset": 0,
        "det_rotation": 0,
        "arc": 2 * np.pi,
        "nangles": 360,
        "DSO": n*20.,
        "DSD": n*20. + n * 2.,
        "COR": 0,
    }


def setup_astra_forward_2d(n):
    cfg = config(n)
    vol_geom = astra.create_vol_geom(n, n)
    det_width = cfg["det_spacing"]
    det_count = cfg["det_shape"]
    angles = np.linspace(0, cfg["arc"], cfg["nangles"], endpoint=False)
    DSO = cfg["DSO"]
    DSD = cfg["DSD"]

    proj_geom = astra.create_proj_geom(
        'fanflat', det_width, det_count, angles, DSO, DSD - DSO)

    volume = np.flip(shepp_logan(cfg["vol_shape"])).copy()

    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

    return {
        "volume": volume,
        "proj_id": proj_id,
    }


def setup_astra_backward_2d(n):
    cfg = config(n)
    vol_geom = astra.create_vol_geom(n, n)
    det_width = cfg["det_spacing"]
    det_count = cfg["det_shape"]
    angles = np.linspace(0, cfg["arc"], cfg["nangles"], endpoint=False)
    DSO = cfg["DSO"]
    DSD = cfg["DSD"]

    proj_geom = astra.create_proj_geom(
        'fanflat', det_width, det_count, angles, DSO, DSD - DSO)

    volume = np.flip(shepp_logan(cfg["vol_shape"])).copy()

    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
    _, sino = astra.create_sino(volume, proj_id)

    return {
        "sino": sino,
        "proj_id": proj_id,
    }


def kernel_astra_forward_2d(volume, proj_id):
    _, sino = astra.create_sino(volume, proj_id)
    return sino


def kernel_astra_backward_2d(sino, proj_id):
    _, bp = astra.create_backprojection(sino, proj_id)
    return bp


# forward = [
#     (kernel_astra_forward_2d, setup_astra_forward_2d(2**i),
#      f"forward 2d {2**i:>4}",  "astra", config(2**i)) for i in range(5, 12)
# ]
#
# backward = [
#     (kernel_astra_backward_2d, setup_astra_backward_2d(2**i),
#      f"backward 2d {2**i:>4}", "astra", config(2**i)) for i in range(5, 12)
# ]

forward = [
    (kernel_astra_forward_2d, setup_astra_forward_2d,
     "forward 2d",  "astra", config)
]

backward = [
    (kernel_astra_backward_2d, setup_astra_backward_2d,
     "backward 2d", "astra", config)
]

__benchmarks__ = forward + backward
