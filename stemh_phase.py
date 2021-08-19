# base python modules
import glob
import os
import numpy as np
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py

# NCEM + Molecular Foundry modules
import stempy.io as stio
from stempy.io import sparse_array
import stempy.image as stim

# our modules
# import fft_tools
import data_selection as ds

dataPath = '/Users/andrewducharme/Documents/Data/ercius_210420/data_scan332_th4.0_electrons.h5'
savePath = '/Users/andrewducharme/Documents/Sparse Code/'
vacuumPath = '/Users/andrewducharme/Documents/Data/ercius_210420/Other Files/scan332_vacuum.dm4'

# extract specimen information
sa = sparse_array.SparseArray.from_hdf5(dataPath)
sa = sa[:, :-1]  # cut off flyback frames

scan_row_num = sa.shape[0]  # same as sp.attrs['Ny'] in hdf5 file metadata
scan_col_num = sa.shape[1]  # same as sp.attrs['Nx'] - 1, since flyback is removed
frame_row_num = sa.shape[2]
frame_col_num = sa.shape[3]

# figure out how much cropping of the Fourier transform is needed once looping
crop_amount = ds.select_FT_signal(sa.data, frame_row_num, frame_col_num, 0.1)
crop_start = crop_amount
row_crop_end = frame_row_num - crop_amount
col_crop_end = frame_col_num - crop_amount

# find amplitude at every scan position
ampMap = ds.find_amplitude(sa)

length = 75  # standardizes area cropped around peaks for analysis
map = np.zeros(num_frames, dtype=np.complex64)

# create vacuum kernel
vacuum_kernel = ds.calc_vacuum_kernel(vacuumPath)
kernel_peak = ds.grab_square_sect(vacuum_kernel, length)

for i, frame in enumerate(raveled_sa):
    if not frame.any():
        map[i] = 0
        continue

    cropped_frame = frame[crop_start:row_crop_end, crop_start:col_crop_end]
    fft = np.fft.fftshift(np.fft.rfft2(cropped_frame))
    abs_fft = abs(fft)

    # find 2 largest magnitude points in the fft (0th and 1st order peaks)
    peaks = ds.fft_find_peaks(abs_fft, 2)
    zeroth_order = peaks[0, 1:].astype(int)
    first_order = peaks[1, 1:].astype(int)
    # print(peaks)

    fourier_space_peak = ds.grab_square_sect(fft, length, first_order)

    # * is element-wise multiplication
    t_temp = np.sum(kernel_peak * fourier_space_peak)

    map[i] = np.angle(t_temp)

phaseMap = map.reshape(scan_row_num, scan_col_num)
        
fig, ax = plt.subplots(1,2)
ax[0].imshow(np.abs(phaseMap))
ax[1].imshow(ampMap, norm=LogNorm())
plt.show()
plt.close('all')

# save the data to npy files for later analysis instead of running this long code every time prior to doing what you want
np.save(savePath + 'phaseMap', phaseMap)
np.save(savePath + 'ampMap', ampMap)
