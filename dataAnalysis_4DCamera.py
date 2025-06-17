## accelerated version of analysis method used in Yasin et al. APL 133, 23102 (2018)
## and Harvey et al. Phys. Rev. Applied 10, 061001 (2018).

## This process would optimally use the 0th order Fourier peak of a frame
## where all probes go through vacuum. this can remove geometric phase. 
## the part of this script that would set this up does not compile
## however, the script as a whole should still work

## This process assumes the value the FFT produces as the phase is accurate,
## which is not true due to the analogue of leakage for phase
 

# base python modules
import numpy as np
import time
import matplotlib.pyplot as plt
import pyfftw
from matplotlib.colors import LogNorm
from multiprocessing import cpu_count

# NCEM + Molecular Foundry modules
from stempy.io import sparse_array

# our module
import stemh_tools as st

dataPath = '/Users/andrewducharme/Documents/Data/ercius_210420/data_scan332_th4.0_electrons.h5'
savePath = '/Users/andrewducharme/Documents/Sparse Code/'

# extract specimen information
# open 4dstem data from h5 file
sa = sparse_array.SparseArray.from_hdf5(dataPath)
sa = sa[:, :-1]  # cut off flyback frames

scan_row_num = sa.shape[0]  # same as scan_positions.attrs['Ny'] in hdf5 file metadata
scan_col_num = sa.shape[1]  # same as sp.attrs['Nx'] - 1, since flyback is removed
frame_row_num = sa.shape[2]
frame_col_num = sa.shape[3]
# numpy likes (# of rows, # of columns), but stempy likes (# of columns, # of rows)

ampMap = st.calc_amplitude(sa)

ravelled_sa = sa.ravel_scans()  # take sparse 4Dstem data and flatten into 3D dense
num_frames = ravelled_sa.shape[0]

# this section computes 2 quantities we'll be using in every step of the loop
# do so with a representative frame in the dataset
# finds 1. location of the Fourier peaks and 2. how large a square will be selected around first order peak.

rep_frame = ravelled_sa[num_frames // 2]  # get representative frame from center of measurement
rep_fft = st.fftw2D(rep_frame)

fft_peaks = st.fft_find_peaks(rep_fft, 2)  # find two highest magnitude peaks in rep_fft

first_order = fft_peaks[1, 1:]  # location of first order peak
selection_size = st.calc_box_size(fft_peaks)

# initialize array to store values through loop
phaseMap = np.zeros(num_frames, dtype=np.complex64)

# # create vacuum kernel
# vacuum_kernel = st.kernelize_vacuum_scan(ravelled_sa[50*50])  # human input needed to select what is truly a vacuum frame
# kernel_peak = st.grab_square_box(vacuum_kernel, selection_size)  # functionally a Dirac delta

# setting up pyfftw numpy interface
pyfftw.config.NUM_THREADS = cpu_count()
pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'
pyfftw.interfaces.cache.enable()

start = time.time()

for i, frame in enumerate(ravelled_sa):
    if i % 5000 == 0:
        print(i)
    if not frame.any():
        phaseMap[i] = 0
        continue

    # copy frame data into bit alligned float32 array
    input = np.empty(frame.shape, dtype='float32')
    input[:] = frame
    
    ft = pyfftw.interfaces.numpy_fft.rfft2(input)  # take Fourier transform of the full frame
    ft = pyfftw.interfaces.numpy_fft.fftshift(ft)

    # fourier_space_peak = st.grab_square_box(ft, selection_size, first_order)  # select the area around the first peak
    # phaseMap[i] = np.sum(kernel_peak * fourier_space_peak)  # Integrate kernel and first order peak product
    phaseMap[i] = ft[first_order[0], first_order[1]]

phaseMap = np.angle(phaseMap)  # phase computation 
phaseMap = phaseMap.reshape(scan_row_num, scan_col_num)

end = time.time()

print("Total time (s): " + str(end - start))
print(str((end - start) / scan_row_num / scan_col_num))
print('1024 x 1024 time (min): ' + str((end - start) / scan_row_num / scan_col_num * 1024 * 1024 / 60))

fig, ax = plt.subplots(1,2)
ax[0].imshow(phaseMap)
ax[1].imshow(ampMap)
plt.show()
plt.close('all')

# np.save(savePath + 'phaseMap', phaseMap)
# np.save(savePath + 'ampMap', ampMap)
