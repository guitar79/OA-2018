import numpy as np
import matplotlib.pyplot as plt

# Load data
# NumPy automatically skips the commented rows.
F450 = np.loadtxt('hst_11233_06_wfpc2_f450w_wf/hst_11233_06_wfpc2_f450w_wf_daophot_trm.cat') 
F814 = np.loadtxt('hst_11233_06_wfpc2_f814w_wf/hst_11233_06_wfpc2_f814w_wf_daophot_trm.cat')

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

# Plot HR Diagram (Color Index - Instrument Magnitude)
# axis lable, scale, etc....
plt.plot(color[mask], F450[mask,5], 'o', ms=1, alpha=0.2)
plt.xlabel('Color (F450W - F814W)')
plt.ylabel('Instrument Magnitude (F450W)')
plt.gca().invert_yaxis()

#show the Diagram
plt.show()

#save the Diagram


