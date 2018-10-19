

import numpy as np
import matplotlib.pyplot as plt

# Load data
# NumPy automatically skips the commented rows.
F450 = np.loadtxt('hst_11233_06_wfpc2_f450w_wf_daophot_trm.cat') 
F814 = np.loadtxt('hst_11233_06_wfpc2_f814w_wf_daophot_trm.cat')

# get CI value and masking 
mask  = np.zeros(len(F450)).astype(bool)
color = np.zeros(len(F450))

# Get color indices of the "same" stars in two filter images
for i in range(0, len(F450)):
    ID450 = F450[i,4]
    for j in range(0, len(F814)):
        ID814 = F814[j,4]
        if ID450 == ID814:
            mask[i]  = True
            color[i] = F450[i,5] - F814[j,5]
            break # Once the matched item is found from F814, go to next star of F450.

# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: [8.0, 6.0]
print ("Current size:", fig_size)
 
# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

plt.plot(color[mask], F450[mask,5], 'o', ms=2, alpha=0.2)
plt.title('H-R Diagram of M80', fontsize=20)
plt.xlabel('Color Index (F450W - F814W)', fontsize=18)
plt.ylabel('Magnitude (F450W)', fontsize=18)
plt.gca().invert_yaxis()

plt.grid()


#save the Diagram
plt.savefig('H-R Diagram.png', dpi=None, facecolor='w', edgecolor='w', \
        orientation='portrait', papertype=None, format=None, \
        transparent=False, bbox_inches=None, pad_inches=0.1, \
        frameon=None, metadata=None)

plt.show()


