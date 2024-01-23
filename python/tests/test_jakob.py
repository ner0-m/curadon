import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import curadon

filename = "/run/media/david/DATA2/CT_data/Jakob/lungman.npz"

npzfile = np.load(filename)
# sw-based attenuation reco
vol = torch.FloatTensor(npzfile['vol']).cuda()
vol = torch.transpose(vol, 0, 1)
print(vol.shape)
sino = torch.FloatTensor(npzfile['sino']).cuda()

DSO = 570
DSD = 1040

det_shape = np.array(sino.shape[1:])
det_extent = np.array((36.5412085, 952.8929005))
det_spacing = det_extent / det_shape
print(det_shape, det_spacing, det_extent)
# det_extent = det_shape * det_spacing
COR = 4.04

vol_shape = vol.shape
vol_spaing = np.array([0.6] * 3)

nangles = sino.shape[0]
angles = np.linspace(0, 2 * np.pi, nangles, endpoint=False)

geom = curadon.ConeGeometry(DSO=DSO,
                            DSD=DSD,
                            angles=angles,
                            vol_shape=vol_shape,
                            vol_spacing=vol_spaing,
                            det_shape=det_shape,
                            det_spacing=det_spacing,
                            COR=COR)


mysino = curadon.forward(vol, geom)
myvol = curadon.backward(mysino, geom)

# pretty print min, max and mean of mysino and sino, using f-strings
print(
    f"mysino min: {mysino.min():.3f}, max: {mysino.max():.3f}, mean: {mysino.mean():.3f}")
print(
    f"sino   min: {sino.min():.3f}, max: {sino.max():.3f}, mean: {sino.mean():.3f}")


fig, ax = plt.subplots(1, 2)
init = vol.shape[1] // 2
img1 = ax[0].imshow(vol[:, init, :].cpu().detach().numpy(), cmap="gray")
img2 = ax[1].imshow(myvol[:, init, :].cpu().detach().numpy(), cmap="gray")
axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='Volume Slice',
    valmin=0,
    valstep=list(range(0, vol.shape[1])),
    valmax=vol.shape[1],
    valinit=init,
)


def update(val):
    img1.set_data(vol[:, int(val), :].cpu().detach().numpy())
    img2.set_data(myvol[:, int(val), :].cpu().detach().numpy())
    fig.canvas.draw_idle()


freq_slider.on_changed(update)
plt.show()

# fig, ax = plt.subplots(2, 1)
# init = sino.shape[0] // 2
# img1 = ax[0].imshow(mysino[init].cpu().detach().numpy(), cmap="gray")
# img2 = ax[1].imshow(sino[init].cpu().detach().numpy(), cmap="gray")
# axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
# freq_slider = Slider(
#     ax=axfreq,
#     label='Sinogram Slice',
#     valmin=0,
#     valstep=list(range(0, nangles)),
#     valmax=nangles,
#     valinit=init,
# )
#
#
# def update(val):
#     img1.set_data(mysino[int(val)].cpu().detach().numpy())
#     img2.set_data(sino[int(val)].cpu().detach().numpy())
#     fig.canvas.draw_idle()
#
#
# freq_slider.on_changed(update)
# plt.show()

# fig, axes = plt.subplots(1, 2, figsize=(10, 10))
# idx = 400
# axes[0].imshow(mysino[idx].cpu().detach().numpy(), cmap="gray")
# axes[1].imshow(sino[idx * 10].cpu().detach().numpy(), cmap="gray")
# plt.show()
