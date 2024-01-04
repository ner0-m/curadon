import matplotlib.pyplot as plt
import pyelsa as elsa
import numpy as np
import io


volsize = 64
phantom = np.asarray(elsa.phantoms.modifiedSheppLogan([volsize] * 2))
phantom[phantom < 0] = 0

np.set_printoptions(linewidth=200)

f = open("phantom_2d.txt", "w")
f.write("type vol\n")

if(len(phantom.shape) == 2):
    f.write(f"size {volsize} {volsize} 1\n")
else:
    f.write(f"size {volsize} {volsize} {volsize}\n")

bio = io.BytesIO()
np.savetxt(bio, phantom.flatten())
vol_str = bio.getvalue().decode('latin1')

f.write(vol_str)
f.close()

print(np.min(phantom), np.max(phantom))
# plt.imshow(np.array(phantom[volsize // 2]))
# plt.show()
