from torch_radon import FanBeam, Volume2D
import torch_radon
import numpy as np
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
        "dtype": torch.float32,
    }


def setup_torchradon_forward_2d(n):
    cfg = config(n)
    dtype = cfg["dtype"]

    vol_shape = cfg["vol_shape"]
    vol_spacing = cfg["vol_spacing"]
    vol_offset = cfg["vol_offset"]

    det_shape = cfg["det_shape"]
    det_spacing = cfg["det_spacing"]

    DSO = cfg["DSO"]
    DSD = cfg["DSD"]

    angles = np.linspace(0, cfg["arc"], cfg["nangles"], endpoint=False)

    volume = Volume2D(center=vol_offset, voxel_size=vol_spacing)
    volume.set_size(vol_shape[0], vol_shape[1])
    radon = FanBeam(det_count=det_shape,
                    angles=angles,
                    src_dist=DSO,
                    det_dist=DSD-DSO,
                    det_spacing=det_spacing,
                    volume=volume
                    )

    phantom = np.flip(shepp_logan(cfg["vol_shape"])).copy()
    x = torch.from_numpy(phantom).type(dtype).cuda()
    return {"radon": radon, "x": x}


def setup_torchradon_backward_2d(n):
    cfg = config(n)
    dtype = cfg["dtype"]

    vol_shape = cfg["vol_shape"]
    vol_spacing = cfg["vol_spacing"]
    vol_offset = cfg["vol_offset"]

    det_shape = cfg["det_shape"]
    det_spacing = cfg["det_spacing"]

    DSO = cfg["DSO"]
    DSD = cfg["DSD"]

    angles = np.linspace(0, cfg["arc"], cfg["nangles"], endpoint=False)

    volume = Volume2D(center=vol_offset, voxel_size=vol_spacing)
    volume.set_size(vol_shape[0], vol_shape[1])
    radon = FanBeam(det_count=det_shape,
                    angles=angles,
                    src_dist=DSO,
                    det_dist=DSD-DSO,
                    det_spacing=det_spacing,
                    volume=volume
                    )

    phantom = np.flip(shepp_logan(cfg["vol_shape"])).copy()
    vol = torch.from_numpy(phantom).type(dtype).cuda()

    sino = radon.forward(vol)

    return {
        "radon": radon, "sinogram": sino
    }


def kernel_torchradon_forward_2d(radon, x):
    return radon.forward(x)


def kernel_torchradon_backward_2d(radon, sinogram):
    return radon.backward(sinogram)


forward = [
    (kernel_torchradon_forward_2d, setup_torchradon_forward_2d,
     "forward 2d",  "torch-radon", config)
]

backward = [
    (kernel_torchradon_backward_2d, setup_torchradon_backward_2d,
     "backward 2d", "torch-radon", config)
]
__benchmarks__ = forward + backward
