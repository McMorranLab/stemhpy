# base python modules
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# NCEM + Molecular Foundry modules
from stempy.io import sparse_array

# our module
import phase_extraction as pe

dataPath = '/Users/andrewducharme/Documents/Data/ercius_210420/data_scan332_th4.0_electrons.h5'
savePath = '/Users/andrewducharme/Documents/Sparse Code/'
vacuumPath = '/Users/andrewducharme/Documents/Data/ercius_210420/Other Files/scan332_vacuum.dm4'

# extract specimen information
# open 4dstem data from h5 file
sa = sparse_array.SparseArray.from_hdf5(dataPath)
sa = sa[:, :-1]  # cut off flyback frames

scan_row_num = sa.shape[0]  # same as scan_positions.attrs['Ny'] in hdf5 file metadata
scan_col_num = sa.shape[1]  # same as sp.attrs['Nx'] - 1, since flyback is removed
frame_row_num = sa.shape[2]
frame_col_num = sa.shape[3]
# numpy likes (# of rows, # of columns), but stempy likes (# of columns, # of rows)

ravelled_sa = sa.ravel_scans()  # take sparse 4Dstem data and flatten into 3D dense
num_frames = ravelled_sa.shape[0]

# find where the Fourier peaks are and how large selection regions around the first order peak should be
first_order, selection_size = pe.phase_setup(ravelled_sa)

# initialize array to store values through loop
phaseMap = np.zeros(num_frames, dtype=np.complex64)
ampMap = np.zeros((selection_size, selection_size))

# create vacuum kernel
vacuum_kernel = pe.calc_vacuum_kernel(vacuumPath)
kernel_peak = pe.grab_square_sect(vacuum_kernel, selection_size)  # functionally a Dirac delta

start = time.time()

# the forward Fourier transform is the majority of the work+computation time here,
# so it's not worth splitting this into two commands
for i, frame in enumerate(ravelled_sa):
    if i % 5000 == 0:
        print(i)
    if not frame.any():
        phaseMap[i] = 0
        continue

    # working numpy code
    # fft = np.fft.fftshift(np.fft.rfft2(frame))
    # ampMap = ampMap + np.abs(np.fft.ifft2(fourier_space_peak))

    # working fftw code
    ft = pe.fftw2D(frame)  # take Fourier transform of the full frame

    fourier_space_peak = pe.grab_square_sect(ft, selection_size, first_order)  # select the area around the first peak

    # amplitude computation
    ifftw = pe.fftw2D(fourier_space_peak, forward=False)  # inverse Fourier transform area around peak
    ampMap = ampMap + np.abs(ifftw)

    # phase computation 
    t_temp = np.sum(kernel_peak * fourier_space_peak)  # convolve kernel and first order peak (* multiplies elementwise)
    phaseMap[i] = np.angle(t_temp)  

phaseMap = phaseMap.reshape(scan_row_num, scan_col_num)

end = time.time()

print("Total time (s): " + str(end - start))
print(str((end - start) / scan_row_num / scan_col_num))
print('1024 x 1024 time (min): ' + str((end - start) / scan_row_num / scan_col_num * 1024 * 1024 / 60))

fig, ax = plt.subplots(1,2)
ax[0].imshow(np.abs(phaseMap))
ax[1].imshow(ampMap)
plt.show()
plt.close('all')

# np.save(savePath + 'phaseMap', phaseMap)
# np.save(savePath + 'ampMap', ampMap)
