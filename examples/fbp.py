import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import curadon
from phantominator import shepp_logan


def config(n):
    return {
        "vol_shape": np.array([n, n], dtype=np.uint32),
        "vol_spacing": np.array([1, 1], dtype=np.float32),
        "vol_offset": np.array([0, 0], dtype=np.float32),
        "det_shape": int(n * np.sqrt(2)),
        "det_spacing": 1.,
        "det_offset": 0.,
        "det_rotation": 0.,
        "arc": 2 * np.pi,
        "nangles": 360,
        "DSO": n * 20.,
        "DSD": n * 20. + n * 2.,
        "COR": 0,
    }


def _get_filter(size, filter_name, frequency_scaling=1.0):
    W = torch.linspace(-1, 1, size)

    w = torch.abs(W)
    if filter_name == "ram-lak":
        pass
    elif filter_name == "shepp-logan":
        w *= torch.sinc(W / (2 * frequency_scaling))
    elif filter_name == "cosine":
        w *= torch.cos(H * torch.pi / (2 * frequency_scaling))
    elif filter_name == "hamming":
        w *= 0.54 + 0.46 * torch.cos(w * torch.pi / (frequency_scaling))
    elif filter_name == "hann":
        w *= torch.cos(w * torch.pi / (2 * frequency_scaling)) ** 2
    elif filter_name is None:
        return torch.ones(size)
    else:
        raise RuntimeError(f"Unknown Filter {filter_name}")

    indicator = W <= frequency_scaling
    w *= indicator
    return torch.FloatTensor(w).cuda()


def _apply_filter(sinogram, filt):
    filt = torch.tile(filt, (sinogram.shape[0], 1))
    fftsino = torch.fft.fft(sinogram, dim=1)
    projection = torch.fft.fftshift(
        fftsino, dim=0) * torch.fft.fftshift(filt, dim=1)
    return torch.real(torch.fft.ifft(torch.fft.ifftshift(projection, dim=0), dim=1))


def main(n):
    cfg = config(n)

    vol_shape = cfg["vol_shape"]
    vol_spacing = cfg["vol_spacing"]
    vol_offset = cfg["vol_offset"]

    det_shape = cfg["det_shape"]
    det_spacing = cfg["det_spacing"]
    det_offset = cfg["det_offset"]
    nangles = cfg["nangles"]
    angles = np.linspace(0, cfg["arc"], nangles, endpoint=False)

    # It's totally fine to pass an array to describe more complex trajectories with
    # varying distances for each angle
    DSO = np.full(nangles, cfg["DSO"])
    DSD = np.full(nangles, cfg["DSD"])
    geom = curadon.FanGeometry(
        DSD=DSD, DSO=DSO, angles=angles, vol_shape=vol_shape, vol_spacing=vol_spacing, det_shape=det_shape, det_spacing=det_spacing,
        vol_prec=torch.finfo(torch.float32).bits,
        sino_prec=torch.finfo(torch.float32).bits,
    )

    phantom = np.flip(shepp_logan(vol_shape)).copy()
    volume = torch.from_numpy(phantom).type(torch.float32).cuda()

    sino = torch.zeros(len(angles), det_shape).type(torch.float32).cuda()

    curadon.forward(volume, sino, geom)

    filt = _get_filter(det_shape, "ram-lak")
    fsino = _apply_filter(sino.type(torch.float32), filt).type(
        torch.float32) * (np.pi / (2 * len(angles)))

    fbp = torch.zeros_like(volume)
    curadon.backward(fsino, fbp, geom)

    # fbp = fbp.cpu().detach().numpy()
    plt.imshow(fbp.cpu().detach().numpy(), cmap="gray")
    # plt.imshow(sino.cpu().detach().numpy(), cmap="gray")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)

    args = parser.parse_args()

    main(args.size)
