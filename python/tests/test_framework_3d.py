import perftester as pt
import perfplot
import matplotlib.pyplot as plt
import numpy as np
import curadon
import torch
import astra
from torch_radon import FanBeam, Volume2D

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


nangles = 360


def forward_curadon(n):
    vol_shape = np.array([n, n], dtype=np.uint32)

    shepp_logan = np.asarray(elsa.phantoms.modifiedSheppLogan(vol_shape))
    volume = torch.from_numpy(shepp_logan).type(torch.float32).cuda()

    det_shape = int(n * np.sqrt(2))

    angles = np.linspace(0, 2 * np.pi, nangles, endpoint=False)
    sinogram = torch.zeros(len(angles), det_shape).type(torch.float32).cuda()
    DSO = n * 20.
    DSD = DSO + n * 2.
    COR = 0.

    geom = curadon.FanGeometry(DSD, DSO, angles, vol_shape, det_shape)

    sino = curadon.forward(volume, geom)
    bp = curadon.backward(sino, geom)
    return sino.cpu().detach().numpy(), bp.cpu().detach().numpy()


def forward_elsa(n):
    vol_shape = np.array([n, n], dtype=np.uint32)
    vol_spacing = np.array([1, 1], dtype=np.float32)
    vol_offset = np.array([0, 0], dtype=np.float32)

    det_shape = int(n * np.sqrt(2))
    det_spacing = 1
    det_offset = 0

    shepp_logan = elsa.phantoms.modifiedSheppLogan(vol_shape)
    vol_desc = shepp_logan.getDataDescriptor()
    angles = np.linspace(0, 360, nangles, endpoint=False)

    DSO = n * 20.
    DSD = DSO + n * 2.

    sino_desc = elsa.CircleTrajectoryGenerator.trajectoryFromAngles(
        angles,
        vol_desc,
        DSO,
        DSD - DSO,
        [det_offset],
        vol_offset,
        [det_shape],
        [det_spacing],
    )
    sino = elsa.DataContainer(np.zeros((len(angles), det_shape)), sino_desc)
    A = elsa.JosephsMethodCUDA(vol_desc, sino_desc)

    A.apply(shepp_logan, sino)
    bp = A.applyAdjoint(sino)
    return np.asarray(sino), np.asarray(bp)


def forward_torchradon(n):
    vol_shape = np.array([n, n], dtype=np.uint32)
    vol_spacing = np.array([1, 1], dtype=np.float32)
    vol_offset = np.array([0, 0], dtype=np.float32)

    det_shape = int(n * np.sqrt(2))
    det_spacing = 1

    DSO = n * 20.
    DSD = DSO + n * 2.

    angles = np.linspace(0, 2 * np.pi, nangles, endpoint=False)

    radon = FanBeam(det_count=det_shape,
                    angles=angles,
                    src_dist=DSO,
                    det_dist=DSD-DSO,
                    det_spacing=det_spacing,
                    volume=Volume2D(voxel_size=vol_spacing)
                    )

    img = np.array(elsa.phantoms.modifiedSheppLogan(vol_shape))
    volume = torch.FloatTensor(img).cuda()
    sino = radon.forward(volume)
    bp = radon.backward(sino)
    return sino.cpu().detach().numpy(), bp.cpu().detach().numpy()


def forward_astra(n):
    vol_geom = astra.create_vol_geom(n, n)
    det_width = 1.
    det_count = int(n * np.sqrt(2))
    angles = np.linspace(0, 2 * np.pi, nangles, endpoint=False)
    DSO = n * 20.
    DSD = DSO + n * 2.

    proj_geom = astra.create_proj_geom(
        'fanflat', det_width, det_count, angles, DSO, DSD - DSO)

    img = np.array(elsa.phantoms.modifiedSheppLogan((n, n)))

    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
    sinogram_id, sinogram = astra.create_sino(img, proj_id)
    bp_id, bp = astra.create_backprojection(sinogram, proj_id)

    return sinogram, bp


n = 256
fp_curadon, bp_curadon = forward_curadon(n)
fp_elsa, bp_elsa = forward_elsa(n)
fp_torchradon, bp_torchradon = forward_torchradon(n)
fp_astra, bp_astra = forward_astra(n)


fig, ax = plt.subplots(2, 4)
ax[0][0].imshow(fp_curadon, cmap="gray")
ax[0][0].set_title("fp curadon")
ax[0][0].grid(False)
ax[0][1].imshow(np.abs(fp_curadon - fp_elsa), cmap="gray")
ax[0][1].set_title("fp elsa")
ax[0][1].grid(False)
ax[0][2].imshow(np.abs(fp_curadon - fp_torchradon), cmap="gray")
ax[0][2].set_title("fp torch-radon")
ax[0][2].grid(False)
ax[0][3].imshow(np.abs(fp_curadon - fp_astra), cmap="gray")
ax[0][3].set_title("fp torch-radon")
ax[0][3].grid(False)

ax[1][0].imshow(bp_curadon, cmap="gray")
ax[1][0].set_title("bp curadon")
ax[1][0].grid(False)
ax[1][1].imshow(np.abs(bp_curadon - bp_elsa), cmap="gray")
ax[1][1].set_title("bp elsa")
ax[1][1].grid(False)
ax[1][2].imshow(np.abs(bp_curadon - bp_torchradon), cmap="gray")
ax[1][2].set_title("bp torch-radon")
ax[1][2].grid(False)
ax[1][3].imshow(np.abs(bp_curadon - bp_astra), cmap="gray")
ax[1][3].set_title("bp torch-radon")
ax[1][3].grid(False)

print(f"fp curadon:    min = {np.min(fp_curadon):>10.8}, max: {np.max(fp_curadon):>10.8}, mean: {np.mean(fp_curadon):>10.8}, median: {np.median(fp_curadon):>10.8}")
print(f"fp elsa:       min = {np.min(fp_elsa):>10.8}, max: {np.max(fp_elsa):>10.8}, mean: {np.mean(fp_elsa):>10.8}, median: {np.median(fp_elsa):>10.8}")
print(f"fp torchradon: min = {np.min(fp_torchradon):>10.8}, max: {np.max(fp_torchradon):>10.8}, mean: {np.mean(fp_torchradon):>10.8}, median: {np.median(fp_torchradon):>10.8}")
print(f"fp astra:      min = {np.min(fp_astra):>10.8}, max: {np.max(fp_astra):>10.8}, mean: {np.mean(fp_astra):>10.8}, median: {np.median(fp_astra):>10.8}")
print()
print(f"bp curadon:    min = {np.min(bp_curadon):>10.8}, max: {np.max(bp_curadon):>10.8}, mean: {np.mean(bp_curadon):>10.8}, median: {np.median(bp_curadon):>10.8}")
print(f"bp elsa:       min = {np.min(bp_elsa):>10.8}, max: {np.max(bp_elsa):>10.8}, mean: {np.mean(bp_elsa):>10.8}, median: {np.median(bp_elsa):>10.8}")
print(f"bp torchradon: min = {np.min(bp_torchradon):>10.8}, max: {np.max(bp_torchradon):>10.8}, mean: {np.mean(bp_torchradon):>10.8}, median: {np.median(bp_torchradon):>10.8}")
print(f"bp astra:      min = {np.min(bp_astra):>10.8}, max: {np.max(bp_astra):>10.8}, mean: {np.mean(bp_astra):>10.8}, median: {np.median(bp_astra):>10.8}")

plt.show()
