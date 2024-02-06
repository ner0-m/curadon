import pytest
import numpy as np
import imagehash
from PIL import Image
import torch
import curadon
import matplotlib.pyplot as plt

import pyelsa as elsa
elsa.logger_pyelsa_generators.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_line_search.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_projectors.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_proximal_operators.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_functionals.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_io.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_operators.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_projectors_cuda.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_solvers.setLevel(elsa.LogLevel.OFF)


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

    return np.array(elsa.analyticalSheppLogan(voldesc, detdesc))


def forward_curadon(n):
    cfg = config(n)
    vol_shape = cfg["vol_shape"]

    shepp_logan = np.asarray(elsa.phantoms.modifiedSheppLogan(vol_shape))
    volume = torch.from_numpy(shepp_logan).type(torch.float32).cuda()

    det_shape = cfg["det_shape"]

    nangles = cfg["nangles"]
    angles = np.linspace(0, cfg["arc"], nangles, endpoint=False)
    sinogram = torch.zeros(len(angles), det_shape).type(torch.float32).cuda()
    DSO = cfg["DSO"]
    DSD = cfg["DSD"]

    geom = curadon.FanGeometry(DSD, DSO, angles, vol_shape, det_shape)

    sino = curadon.forward(volume, geom)

    return sino.cpu().detach().numpy()


def distance(h1, h2):
    return float(h1 - h2) / len(h1.hash)**2


@pytest.mark.parametrize("n", [128, 256, 512, 1024])
def test_forward(n: int):

    # make this configurabe
    hash_fn = imagehash.phash

    # analytical sinogram
    sino = create_sino(n) * np.pi / (2 * config(n)["nangles"])
    sino_hash = hash_fn(Image.fromarray(sino))

    # forward projection by curadon
    fp = forward_curadon(n) * np.pi / (2 * config(n)["nangles"])
    fp_hash = hash_fn(Image.fromarray(fp))

    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(sino, cmap="gray")
    # ax[1].imshow(fp, cmap="gray")
    # ax[2].imshow(np.abs(fp - sino), cmap="gray")
    # plt.show()

    # fp is an numpy array,
    assert fp.shape == sino.shape

    assert fp.min() == pytest.approx(
        sino.min(), rel=0.01), "Min value is not close to the analytical sinograms min value"
    assert fp.max() == pytest.approx(sino.max(),
                                     rel=0.05), "Max value is not close to the analytical sinograms max value"
    assert fp.mean() == pytest.approx(sino.mean(),
                                      rel=0.05), "Mean value is not close to the analytical sinograms mean value"

    # Test if the forward projection is close to the analytical sinogram
    msg1 = f"Distance of hash is too large: {distance(sino_hash, fp_hash)}"
    assert distance(sino_hash, fp_hash) <= 0.07, msg1

    diff_img = np.abs(fp - sino)

    assert np.min(diff_img) == pytest.approx(0, rel=0.001)
    assert np.max(diff_img) <= 1.5

    img = Image.fromarray(diff_img)

    new_diff_hash = hash_fn(img)

    # Test if the difference between the forward projection and the analytical
    # sinogram is close to some original difference
    msg2 = f"Distance of absolute difference is too large {distance(sino_hash, new_diff_hash)}"
    assert distance(imagehash.hex_to_hash("0000000000000000"),
                    new_diff_hash) <= 0.01, msg2
