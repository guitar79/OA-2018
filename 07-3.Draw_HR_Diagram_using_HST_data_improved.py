import numpy as np
import matplotlib.pyplot as plt

ch1 = 'F450'
ch2 = 'F606'
ch1_file = 'hst_09244_ps_wfpc2_f450w_wf_daophot_trm.cat'
ch2_file = 'hst_09244_ps_wfpc2_f606w_wf_daophot_trm.cat'

object_name = 'M13'

# Load data
# NumPy automatically skips the commented rows.
ch1_data = np.loadtxt(ch1_file) 
ch2_data = np.loadtxt(ch2_file) 

# get CI value and masking 
mask  = np.zeros(len(ch1_data)).astype(bool)
color = np.zeros(len(ch1_data))

# Get color indices of the "same" stars in two filter images
for i in range(0, len(ch1_data)):
    ID_ch1 = ch1_data[i,4]
    for j in range(0, len(ch2_data)):
        ID_ch2 = ch2_data[j,4]
        if ID_ch1 == ID_ch2 :
            mask[i]  = True
            color[i] = ch1_data[i,5] - ch2_data[j,5]
            break # Once the matched item is found from F814, go to next star of F450.

# Plot HR Diagram (Color Index - Instrument Magnitude)
# axis lable, scale, etc....


# Get current size
fig_size = plt.rcParams["figure.figsize"]
print ("Current size:", fig_size)
# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

plt.plot(color[mask], ch1_data[mask,5], 'o', ms=2, alpha=0.2)
plt.title('H-R Diagram of '+object_name, fontsize=20 )
plt.xlabel('Color Index ('+ ch1 + '-' + ch2 + ')', fontsize=18)
plt.ylabel('Magnitude ('+ ch1 + ')', fontsize=18)
plt.gca().invert_yaxis()

plt.grid()


#show the Diagram
plt.show()

plt.savefig('H-R Diagram of %s.png', dpi=None, facecolor='w', edgecolor='w', \
        orientation='portrait', papertype=None, format=None, \
        transparent=False, bbox_inches=None, pad_inches=0.1, \
        frameon=None, metadata=None % (object_name))

