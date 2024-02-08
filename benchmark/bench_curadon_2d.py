import numpy as np
import curadon
import torch
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
        "dtype": torch.float16,
    }


def setup_curadon_forward_2d(n):
    cfg = config(n)
    dtype = cfg["dtype"]

    phantom = np.flip(shepp_logan(cfg["vol_shape"])).copy()
    volume = torch.from_numpy(phantom).type(dtype).cuda()

    det_shape = cfg["det_shape"]
    angles = np.linspace(0, cfg["arc"], cfg["nangles"], endpoint=False)
    DSO = cfg["DSO"]
    DSD = cfg["DSD"]

    geom = curadon.FanGeometry(
        DSD, DSO, angles, cfg["vol_shape"], det_shape,
        vol_dtype=dtype,
        sino_dtype=dtype,
    )
    sino = torch.zeros(cfg["nangles"], det_shape).type(dtype).cuda()

    return {
        "volume": volume, "geom": geom, "sinogram": sino
    }


def setup_curadon_backward_2d(n):
    cfg = config(n)
    dtype = cfg["dtype"]

    angles = np.linspace(0, cfg["arc"], cfg["nangles"], endpoint=False)
    DSO = cfg["DSO"]
    DSD = cfg["DSD"]

    geom = curadon.FanGeometry(
        DSD, DSO, angles, cfg["vol_shape"], cfg["det_shape"],
        vol_dtype=dtype,
        sino_dtype=dtype,
    )

    phantom = np.flip(shepp_logan(cfg["vol_shape"])).copy()
    vol = torch.from_numpy(phantom).type(dtype).cuda()
    sino = torch.zeros(cfg["nangles"], cfg["det_shape"]).type(dtype).cuda()
    curadon.forward(vol, geom, sino)

    return {
        "volume": vol, "geom": geom, "sinogram": sino
    }


def kernel_curadon_forward_2d(volume, geom, sinogram, launch_config=None):
    if launch_config:
        geom.plan.forward_block_x = launch_config[0]
        geom.plan.forward_block_y = launch_config[1]

    curadon.forward(volume, geom, sinogram=sinogram)
    return sinogram


def kernel_curadon_backward_2d(sinogram, geom, volume, launch_config=None):
    if launch_config:
        geom.plan.forward_block_x = launch_config[0]
        geom.plan.forward_block_y = launch_config[1]

    curadon.backward(sinogram, geom, volume=volume)
    return volume


forward = [
    (kernel_curadon_forward_2d, setup_curadon_forward_2d,
     "forward 2d",  "curadon", config)
]

backward = [
    (kernel_curadon_backward_2d, setup_curadon_backward_2d,
     "backward 2d", "curadon", config)
]

__benchmarks__ = forward + backward
