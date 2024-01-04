import matplotlib.pyplot as plt
import pyelsa as elsa
import numpy as np
import io


volsize = 64
detsize = 64 + 28
DSO = 500
DSD = DSO + 100
# angles = np.linspace(0, 360, 32 * 3, endpoint=False)
angles = np.linspace(0, 360,  360, endpoint=False)
voldesc = elsa.VolumeDescriptor([volsize] * 3)
detdesc = elsa.CircleTrajectoryGenerator.trajectoryFromAngles(
    angles, voldesc, DSO, DSD - DSO, detectorSize=[detsize] * 2)

sino = np.array(elsa.analyticalSheppLogan(voldesc, detdesc))

np.set_printoptions(linewidth=200)

f = open("demofile2.txt", "w")
f.write("type sino\n")
f.write(f"size {detsize} {detsize}\n")
f.write(f"nangles {len(angles)}\n")

f.write(f"angles ")
for angle in angles:
    f.write(f"{angle} ")
f.write("\n")
f.write(f"DSO {DSO}\n")
f.write(f"DSD {DSD}\n")

bio = io.BytesIO()
np.savetxt(bio, sino.flatten())
sino_str = bio.getvalue().decode('latin1')

f.write(sino_str)
f.close()

# plt.imshow(np.array(sino[0]))
# plt.show()
