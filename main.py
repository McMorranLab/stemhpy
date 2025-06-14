"""
Levi Brown 

This module is just an example of how to use the fft_tools.py and frame_analysis.py modules

main() computes the phase and amplitude of each frame, and uses parallelization as an approach
to decrease runtime in the main loop.
"""
import numba
import numpy as np
from stempy.io.sparse_array import SparseArray
from frame_analysis import *
import timeit
import multiprocessing as mp
# TODO: Fully annotate main() and clean it up
# TODO: Update Andrew's files from github

def main():

    dataPath = 'data/data_scan18_th4.0_electrons.h5'
    savePath = 'results'
    vacuumPath = 'data/scan332_vacuum.dm4'

    # extract specimen information
    # open 4dstem data from h5 file
    sa = SparseArray.from_hdf5(dataPath)
    sa = sa[:, :-1]  # cut off flyback frames

    scan_row_num = sa.shape[0]  # same as scan_positions.attrs['Ny'] in hdf5 file metadata
    scan_col_num = sa.shape[1]  # same as sp.attrs['Nx'] - 1, since flyback is removed
    frame_row_num = sa.shape[2]
    frame_col_num = sa.shape[3]

    ravelled_sa = sa.ravel_scans()  # take sparse 4Dstem data and flatten into 3D dense
    num_frames = ravelled_sa.shape[0]

    # flatten and store frame indices for e_count functions
    data = sa.data
    f_index = frame_index(data)
    data_flat = np.concatenate(data).astype('u4')

    # compile various functions with numba
    compile_crop = crop_vacuum(sa.data[0])
    compile_ecf = elec_count_frame(data_flat, f_index, (0,), np.array([0,576,0,576]))

    crop_vals = []

    # find the crop vals of each frame
    for frame in sa.data:
        crop_vals.append(crop_vacuum(frame))

    # find where the Fourier peaks are and how large selection regions around the first order peak should be
    first_order, selection_size = pe.phase_setup(ravelled_sa, crop=False, rfft=True)

    # create vacuum kernel
    vacuum_kernel = pe.calc_vacuum_kernel(vacuumPath, crop=False)
    kernel_peak = pe.grab_square_sect(vacuum_kernel, selection_size)  # functionally a Dirac delta

    start = timeit.default_timer()

    # main loop, computes an fft and stores 1st order peak of each frame
    pool = mp.Pool(processes=mp.cpu_count())
    frame_vals = pool.starmap(analyze_frame,
                            [(data_flat, f_index, frame, crop_vals[frame],
                              selection_size, first_order, kernel_peak) for frame in range(num_frames)])
    amp_vals = np.array([np.array(amp, dtype=object) for amp in frame_vals])[:, 0]
    phase_vals = np.array([np.array(phase, dtype=object) for phase in frame_vals])[:, 1]
    ampMap = sum(amp_vals)
    phaseMap = np.array([phase for phase in phase_vals], dtype=np.complex64)
    phaseMap = phaseMap.reshape(scan_row_num, scan_col_num)
    pool.close()

    end = timeit.default_timer()

    print("Total time (s): " + str(end - start))
    print(str((end - start) / scan_row_num / scan_col_num))
    print('1024 x 1024 time (min): ' + str((end - start) / scan_row_num / scan_col_num * 256 * 256 / 60))

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(np.abs(phaseMap))
    ax[1].imshow(ampMap)
    plt.show()


if __name__ == "__main__":
    main()
