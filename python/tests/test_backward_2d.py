import pytest
import numpy as np
import imagehash
from PIL import Image
import torch
import curadon
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
        "DSO": n * 20.,
        "DSD": n * 20. + n * 2.,
        "COR": 0,
    }


def create_shepp_logan(n):
    return np.flip(shepp_logan(config(n)["vol_shape"])).copy()


def _apply_filter(sinogram, filt):
    filt = torch.tile(filt, (sinogram.shape[0], 1))
    fftsino = torch.fft.fft(sinogram, dim=1)
    projection = torch.fft.fftshift(
        fftsino, dim=0) * torch.fft.fftshift(filt, dim=1)
    return torch.real(torch.fft.ifft(torch.fft.ifftshift(projection, dim=0), dim=1))


def backward_curadon(n):
    cfg = config(n)
    vol_shape = cfg["vol_shape"]

    det_shape = cfg["det_shape"]

    nangles = cfg["nangles"]
    angles = np.linspace(0, cfg["arc"], nangles, endpoint=False)
    DSO = cfg["DSO"]
    DSD = cfg["DSD"]

    geom = curadon.FanGeometry(DSD, DSO, angles, vol_shape, det_shape)

    volume = torch.from_numpy(create_shepp_logan(n)).to(torch.float32).cuda()
    sino = curadon.forward(volume, geom)
    filt = torch.abs(torch.linspace(-1, 1, det_shape).cuda())  # Ram-Lak filter
    fsino = _apply_filter(sino, filt) * (np.pi / (2 * nangles))
    fbp = curadon.backward(fsino, geom)

    return fbp.cpu().detach().numpy()


def distance(h1, h2):
    return float(h1 - h2) / len(h1.hash)**2


@pytest.mark.parametrize("n", [128, 256, 512, 1024])
def test_backward(n: int):
    hash_fn = imagehash.phash

    # Original shepp-logan phantom
    volume = create_shepp_logan(n)
    vol_hash = hash_fn(Image.fromarray(volume))

    # FBP, with back projection by curadon
    fbp = backward_curadon(n)
    fbp_hash = hash_fn(Image.fromarray(fbp))

    assert fbp.shape == volume.shape

    # Assert if the filtered backprojection is somewhat close to the original
    # volume (We use enough angles, so that should be fine!)
    dist = distance(vol_hash, fbp_hash)
    msg1 = f"Distance of hash is too large: {dist}"
    assert dist < 0.11, msg1
