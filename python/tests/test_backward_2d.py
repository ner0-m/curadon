import pytest
import matplotlib.pyplot as plt
import numpy as np
import imagehash
from PIL import Image
import pyelsa as elsa
import torch
import curadon

elsa.logger_pyelsa_generators.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_line_search.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_projectors.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_proximal_operators.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_functionals.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_io.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_operators.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_projectors_cuda.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_solvers.setLevel(elsa.LogLevel.OFF)


def normalize(x):
    return ((x - x.min()) * (1/(x.max() - x.min()) * 255)).astype('uint8')


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
        # "nangles": 720,
        "nangles": 256 * 2,
        "DSO": n*20.,
        "DSD": n*20. + n * 2.,
        "COR": 0,
    }


def create_sino(n):
    cfg = config(n)
    det_shape = cfg["det_shape"]
    det_spacing = cfg["det_spacing"]
    det_offset = cfg["det_offset"]

    angles = np.linspace(0, cfg["arc"] * 180. / np.pi,
                         cfg["nangles"], endpoint=False)

    DSO = cfg["DSO"]
    DSD = cfg["DSD"]

    voldesc = elsa.VolumeDescriptor([n, n])
    detdesc = elsa.CircleTrajectoryGenerator.trajectoryFromAngles(
        angles,
        voldesc,
        DSO,
        DSD - DSO,
        [det_offset],
        [0, 0],
        [det_shape],
        [det_spacing],
    )

    return torch.from_numpy(np.array(elsa.analyticalSheppLogan(voldesc, detdesc))).cuda()


def create_shepp_logan(n):
    shepp_logan = np.asarray(elsa.phantoms.modifiedSheppLogan([n, n]))
    return shepp_logan


def _get_filter(size, filter_name, frequency_scaling=1.0):
    W = torch.linspace(-1, 1, size)

    w = torch.abs(W)
    if filter_name == "ram-lak":
        pass
    elif filter_name == "shepp-logan":
        w *= torch.sinc(W / (2 * frequency_scaling))
    elif filter_name == "cosine":
        w *= torch.cos(W * torch.pi / (2 * frequency_scaling))
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


def backward_curadon(n):
    cfg = config(n)
    vol_shape = cfg["vol_shape"]

    det_shape = cfg["det_shape"]

    nangles = cfg["nangles"]
    angles = np.linspace(0, cfg["arc"], nangles, endpoint=False)
    DSO = cfg["DSO"]
    DSD = cfg["DSD"]

    geom = curadon.FanGeometry(DSD, DSO, angles, vol_shape, det_shape)

    sino = create_sino(n)
    # Ram-Lak filter
    filt = torch.abs(torch.linspace(-1, 1, det_shape).cuda())
    # filt = _get_filter(det_shape, "shepp-logan")
    fsino = _apply_filter(sino, filt) * (np.pi / (2 * nangles))
    fbp = curadon.backward(fsino, geom)

    return fbp.cpu().detach().numpy()


difference_hashes = {
    128: imagehash.hex_to_hash("c1d39e4cd999386c"),
    256: imagehash.hex_to_hash("e5c79a98cf494c38"),
    512: imagehash.hex_to_hash("c5d39a9ccf590c2c"),
    1024: imagehash.hex_to_hash("e4c39b9ccb494c69"),
    2048: imagehash.hex_to_hash("e4c39b9cce694c29"),
}


def distance(h1, h2):
    return float(h1 - h2) / len(h1.hash)**2

# Only parameterize on size, this enables easy selection with `pytest -k test_backward[128]`


@pytest.mark.parametrize("n", difference_hashes.keys())
def test_backward(n: int):
    diff_hash = difference_hashes[n]

    # make this configurabe
    # hash_fn = imagehash.phash
    # hash_fn = imagehash.whash
    hash_fn = imagehash.phash

    # Original shepp-logan phantom
    volume = create_shepp_logan(n)
    vol_hash = hash_fn(Image.fromarray(volume))

    # FBP, with back projection by curadon
    fbp = backward_curadon(n)
    # Cleanup some stuff in fbp
    fbp[fbp <= 0.1] = 0
    fbp_hash = hash_fn(Image.fromarray(fbp))

    plt.imshow(fbp, cmap="gray")
    plt.colorbar()
    plt.show()

    assert fbp.shape == volume.shape

    new_diff_hash = hash_fn(Image.fromarray(np.abs(fbp - volume)))

    # Assert if the filtered backprojection is somewhat close to the original
    # volume (We use enough angles, so that should be fine!)
    msg1 = f"Distance of hash is too large: {vol_hash - fbp_hash}"
    dist = float(vol_hash - fbp_hash) / len(vol_hash.hash)**2
    assert vol_hash - \
        fbp_hash < 0, f"len: {len(vol_hash.hash)**2}, Distance: {dist}"

    # Assert that the difference is somewhat close to a previously known state
    # msg2 = f"Distance of hash of difference is too large: {diff_hash - new_diff_hash}"
    # assert diff_hash - new_diff_hash <= 3, msg2
