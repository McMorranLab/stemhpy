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
from ncempy.io import dm

# our modules
# import fft_tools
import data_selection as ds

dataPath = '/Users/andrewducharme/Documents/Data/ercius_210420/data_scan332_th4.0_electrons.h5'
savePath = '/Users/andrewducharme/Documents/Sparse Code/'
vacuumPath = '/Users/andrewducharme/Documents/Data/ercius_210420/Other Files/scan332_vacuum.dm4'

# create vacuum kernel
# read vacuum scan from dm4 file
vacuumFile = dm.fileDM(vacuumPath)
vacuum_data = vacuumFile.getDataset(0)['data']
vacuum_fft = np.fft.fftshift(np.fft.fft2(vacuum_data))
kernel = np.conj(vacuum_fft)

# extract specimen information
sa = sparse_array.SparseArray.from_hdf5(dataPath)
sa = sa[:, :-1, :, :]  # cut off flyback frames

scan_x = sa.shape[0]  # same as sp.attrs['Nx'] in hdf5 file metadata
scan_y = sa.shape[1]  # same as sp.attrs['Ny'] - 1, since flyback is removed
frame_x = sa.shape[2]
frame_y = sa.shape[3]

data = sa.data.reshape((scan_y,scan_x))  # numpy likes row x column, stempy is column x row

empty_frame_counter = 0

length = 75  # fixed crop length for peak analysis
ampMap = np.zeros((scan_y, scan_x))
phaseMap = np.zeros((scan_y, scan_x), dtype=np.complex64)

for i in range(scan_x):
    if i // 25 == i / 25:
        print(i)
    for j in range(scan_y):
        frame = data[i, j]

        if frame.size == 0:
            empty_frame_counter = empty_frame_counter + 1
            ampMap[i,j] = 0
            phaseMap[i,j] = 0
            continue

        diffraction_pattern = stim.calculate_sum_sparse(frame, (frame_x, frame_y))
        fft = np.fft.fftshift(np.fft.rfft2(diffraction_pattern))
        abs_fft = np.abs(fft)

        # find 2 largest magnitude points in the fft (0th and 1st order peaks)
        peaks = ds.fft_find_peaks(abs_fft, 2)
        zeroth_order = peaks[0, 1:].astype(int)
        first_order = peaks[1, 1:].astype(int)

        fourier_space_peak = ds.grab_square_sect(fft, length, first_order)
        kernel_peak = ds.grab_square_sect(kernel, length)

        # * is element-wise multiplication
        t_temp = np.sum(kernel_peak * fourier_space_peak)

        # in notation, np.abs(np.sum(diffraction_pattern)), but same as just length of frame
        hx_p = frame.shape[0]
        # t = hx_p * np.exp(1.0j * np.angle(t_temp))

        phaseMap[i,j] = np.angle(t_temp)
        ampMap[i,j] = hx_p

        
fig, ax = plt.subplots(1,2)
ax[0].imshow(np.abs(phaseMap))
ax[1].imshow(ampMap, norm=LogNorm())
plt.show()
plt.close('all')

# save the data to npy files for later analysis instead of running this long code every time prior to doing what you want
np.save(savePath + 'phaseMap', phaseMap)
np.save(savePath + 'ampMap', ampMap)
