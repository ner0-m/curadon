import perftester as pt
import perfplot
import matplotlib.pyplot as plt
import numpy as np
import curadon
import torch
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
    vol_spacing = np.array([1, 1], dtype=np.float32)
    vol_offset = np.array([0, 0], dtype=np.float32)

    shepp_logan = np.asarray(elsa.phantoms.modifiedSheppLogan(vol_shape))
    volume = torch.from_numpy(shepp_logan).type(torch.float32).cuda()

    det_shape = int(n * np.sqrt(2))
    det_spacing = 1
    det_offset = 0
    det_rotation = 0

    angles = np.linspace(0, 2 * np.pi, nangles, endpoint=False)
    sinogram = torch.zeros(len(angles), det_shape).type(torch.float32).cuda()
    DSO = n * 20.
    DSD = DSO + n * 2.
    COR = 0.

    curadon.forward_2d(volume, vol_shape, vol_spacing, vol_offset, sinogram,
                       angles, det_shape, det_spacing, det_offset, det_rotation, DSO, DSD, COR)

    bp = torch.zeros_like(volume)
    curadon.backward_2d(bp, vol_shape, vol_spacing, vol_offset, sinogram,
                        angles, det_shape, det_spacing, det_offset, det_rotation, DSO, DSD, COR)
    return sinogram.cpu().detach().numpy(), bp.cpu().detach().numpy()



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


n = 256
fd_curadon, bp_curadon = forward_curadon(n)
fd_elsa, bp_elsa = forward_elsa(n)
fd_torchradon, bp_torchradon = forward_torchradon(n)

fig, ax = plt.subplots(2, 3)
ax[0][0].imshow(fd_curadon, cmap="gray")
ax[0][0].set_title("fp curadon")
ax[0][0].grid(False)
ax[0][1].imshow(fd_elsa, cmap="gray")
ax[0][1].set_title("fp elsa")
ax[0][1].grid(False)
ax[0][2].imshow(fd_torchradon, cmap="gray")
ax[0][2].set_title("fp torch-radon")
ax[0][2].grid(False)

ax[1][0].imshow(bp_curadon, cmap="gray")
ax[1][0].set_title("bp curadon")
ax[1][0].grid(False)
ax[1][1].imshow(bp_elsa, cmap="gray")
ax[1][1].set_title("bp elsa")
ax[1][1].grid(False)
ax[1][2].imshow(bp_torchradon, cmap="gray")
ax[1][2].set_title("bp torch-radon")
ax[1][2].grid(False)

print(f"fp curadon min: {np.min(fd_curadon)}, max: {np.max(fd_curadon)}, mean: {np.mean(fd_curadon)}, median: {np.median(fd_curadon)}")
print(f"fp elsa min: {np.min(fd_elsa)}, max: {np.max(fd_elsa)}, mean: {np.mean(fd_elsa)}, median: {np.median(fd_elsa)}")
print(f"fp torchradon min: {np.min(fd_torchradon)}, max: {np.max(fd_torchradon)}, mean: {np.mean(fd_torchradon)}, median: {np.median(fd_torchradon)}")

print(f"bp curadon min: {np.min(bp_curadon)}, max: {np.max(bp_curadon)}, mean: {np.mean(bp_curadon)}, median: {np.median(bp_curadon)}")
print(f"bp elsa min: {np.min(bp_elsa)}, max: {np.max(bp_elsa)}, mean: {np.mean(bp_elsa)}, median: {np.median(bp_elsa)}")
print(f"bp torchradon min: {np.min(bp_torchradon)}, max: {np.max(bp_torchradon)}, mean: {np.mean(bp_torchradon)}, median: {np.median(bp_torchradon)}")

plt.show()
