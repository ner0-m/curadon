import numpy as np
import curadon
import torch

import pyelsa as elsa
import matplotlib.pyplot as plt


def test_forward(show=False):

    vol_shape = np.array([64, 64, 64], dtype=np.uint32)
    vol_spacing = np.array([1, 1, 1], dtype=np.float32)
    vol_offset = np.array([0, 0, 0], dtype=np.float32)
    volume = torch.from_numpy(np.asarray(elsa.phantoms.modifiedSheppLogan(
        vol_shape))).type(torch.float32).cuda()

    det_shape = np.array([96, 96], dtype=np.uint32)
    det_spacing = np.array([1, 1], dtype=np.float32)
    det_offset = np.array([0, 0], dtype=np.float32)
    det_rotation = np.array([0, 0, 0], dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    sinogram = torch.zeros(len(angles), *det_shape).type(torch.float32).cuda()
    DSO = 1000.
    DSD = 1150.
    COR = 0.

    curadon.forward_3d(volume, vol_shape, vol_spacing, vol_offset, sinogram,
                       angles, det_shape, det_spacing, det_offset, det_rotation, DSO, DSD, COR)

    if show:
        plt.imshow(sinogram[180, :, :].cpu().numpy(), cmap="gray")
        plt.show()


if __name__ == '__main__':
    test_forward(show=True)
