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
    fsino = _apply_filter(sino, _get_filter(det_shape, "ram-lak")) * (np.pi / (2 * len(nangles)))
    bp = curadon.backward(fsino, geom)
    bp[bp < 0] = 0

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

    npsino = torch.from_numpy(np.asarray(sino)).cuda()
    fsino = _apply_filter(npsino, _get_filter(det_shape, "ram-lak"))
    dc_fsino = elsa.DataContainer(fsino.cpu().detach().numpy(), sino_desc)
    bp = A.applyAdjoint(dc_fsino)
    bp = np.asarray(bp)
    bp[bp < 0] = 0

    return np.array(sino), bp


def forward_torchradon(n):
    vol_shape = np.array([n, n], dtype=np.uint32)
    vol_spacing = np.array([1, 1], dtype=np.float32)
    vol_offset = np.array([0, 0], dtype=np.float32)

    det_shape = int(n * np.sqrt(2))
    det_spacing = 1

    DSO = n * 20.
    DSD = DSO + n * 2.

    angles = np.linspace(0, 2 * np.pi, nangles, endpoint=False)

    volume = Volume2D(voxel_size=vol_spacing)
    volume.set_size(n, n)
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
    fsino = _apply_filter(sino.clone(), _get_filter(det_shape, "ram-lak"))
    bp = radon.backward(fsino)
    bp[bp < 0] = 0

    # sino = curadon.forward(volume, geom)
    # fsino = _apply_filter(sino, _get_filter(det_shape, "ram-lak"))
    # bp = curadon.backward(fsino, geom)
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
    _, sino = astra.create_sino(img, proj_id)
    fsino = _apply_filter(torch.from_numpy(sino).cuda(), _get_filter(
        det_count, "ram-lak")).cpu().detach().numpy()
    _, bp = astra.create_backprojection(fsino, proj_id)
    bp[bp < 0] = 0

    return sino, bp


n = 256
fp_curadon, bp_curadon = forward_curadon(n)
fp_elsa, bp_elsa = forward_elsa(n)
fp_torchradon, bp_torchradon = forward_torchradon(n)
fp_astra, bp_astra = forward_astra(n)

def imshow(ax, img, title=None):
    ax.imshow(img, cmap="gray")
    if title:
        ax.set_title(title)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

fig, ax = plt.subplots(2, 3)
imshow(ax[0][0], fp_curadon, title="curadon")
imshow(ax[0][1], fp_torchradon, title="torchradon")
imshow(ax[0][2], fp_astra, title="astra")
imshow(ax[1][0], bp_curadon)
imshow(ax[1][1], bp_torchradon)
imshow(ax[1][2], bp_astra)
plt.savefig("recos.png", dpi=300, bbox_inches="tight")

# print(f"fp curadon:    min = {np.min(fp_curadon):>10.8}, max: {np.max(fp_curadon):>10.8}, mean: {np.mean(fp_curadon):>10.8}, median: {np.median(fp_curadon):>10.8}")
# print(f"fp elsa:       min = {np.min(fp_elsa):>10.8}, max: {np.max(fp_elsa):>10.8}, mean: {np.mean(fp_elsa):>10.8}, median: {np.median(fp_elsa):>10.8}")
# print(f"fp torchradon: min = {np.min(fp_torchradon):>10.8}, max: {np.max(fp_torchradon):>10.8}, mean: {np.mean(fp_torchradon):>10.8}, median: {np.median(fp_torchradon):>10.8}")
# print(f"fp astra:      min = {np.min(fp_astra):>10.8}, max: {np.max(fp_astra):>10.8}, mean: {np.mean(fp_astra):>10.8}, median: {np.median(fp_astra):>10.8}")
# print()
# print(f"fbp curadon:    min = {np.min(bp_curadon):>10.8}, max: {np.max(bp_curadon):>10.8}, mean: {np.mean(bp_curadon):>10.8}, median: {np.median(bp_curadon):>10.8}")
# print(
#     f"fbp elsa:       min = {np.min(bp_elsa):>10.8}, max: {np.max(bp_elsa):>10.8}, mean: {np.mean(bp_elsa):>10.8}, median: {np.median(bp_elsa):>10.8}")
# print(f"fbp torchradon: min = {np.min(bp_torchradon):>10.8}, max: {np.max(bp_torchradon):>10.8}, mean: {np.mean(bp_torchradon):>10.8}, median: {np.median(bp_torchradon):>10.8}")
# print(f"fbp astra:      min = {np.min(bp_astra):>10.8}, max: {np.max(bp_astra):>10.8}, mean: {np.mean(bp_astra):>10.8}, median: {np.median(bp_astra):>10.8}")

plt.show()
