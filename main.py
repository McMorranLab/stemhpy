"""
This module is just an example of how to use the fft_tools.py and frame_analysis.py modules
"""
from stempy.io.sparse_array import SparseArray
from frame_analysis import *
import timeit
import multiprocessing as mp


def main():
    # stempy hdf5 sparse object
    sparse_330 = SparseArray.from_hdf5('data/data_scan330_th4.0_electrons_centered.h5')
    sa = sparse_330[:, :-1]

    # raw data handling
    data_330 = sparse_330.data
    f_index_330 = frame_index(data_330)
    data_flat_330 = np.concatenate(data_330).astype('u4')

    # construct composite image from raw data
    full_330 = elec_count_full(data_flat_330)

    # crop out vacuum space
    crop = crop_vacuum(full_330, 0.2)
    crop_full_330 = full_330[crop:full_330.shape[0] - crop,
                   crop:full_330.shape[0] - crop]

    # generate an fft of the image and find the peaks
    fft_full_330 = gen_fft(crop_full_330)
    full_peaks = fft_find_peaks(fft_full_330, 10)
    first_peak = np.array([full_peaks[1, 1], full_peaks[1, 2]])

    # mask
    full_mask = fft_mask(fft_full_330, 1, 50, 'square')

    start = timeit.default_timer()
    # main loop, computes an fft and stores 1st order peak of each frame
    pool = mp.Pool(processes=mp.cpu_count())
    fft_peaks = pool.starmap(analyze_frame, [(data_flat_330, f_index_330, frame, crop, first_peak) for frame in range(len(f_index_330))])
    end = timeit.default_timer()
    time = end-start
    pool.close()

    plots(crop_full_330, np.abs(fft_full_330))
    plots(crop_full_330, np.abs(fft_full_330*full_mask))

    plt.show()


if __name__ == "__main__":
    main()

