"""
This module is just an example of how to use the fft_tools.py and frame_analysis.py modules
"""
from stempy.io.sparse_array import SparseArray
from frame_analysis import *
import timeit


def main():
    # stempy hdf5 sparse object
    sparse_330 = SparseArray.from_hdf5('data/data_scan332_th4.5_electrons_centered.h5')
    sa = sparse_330[:, :-1]

    # raw data handling
    data_330 = sparse_330.data
    f_index_330 = frame_index(data_330)
    data_flat_330 = np.concatenate(data_330).astype('u4')

    # construct composite image and fft from raw data
    full_330 = elec_count_full(data_flat_330)
    fft_full_330 = gen_fft(full_330)
    full_peaks = fft_find_peaks(np.abs(fft_full_330), 10)

    # crop out vacuum space
    crop = crop_vacuum(full_330, 0.2)
    cropped_full = full_330[crop:full_330.shape[0] - crop,
                   crop:full_330.shape[0] - crop]

    # index of first order fourier peak
    first_peak = np.array([full_peaks[1, 1], full_peaks[1, 2]])

    fft_peaks = []
    start = timeit.default_timer()
    # main loop, computes an fft and stores 1st order peak of each frame
    for i in np.arange(len(f_index_330)):
        frame = analyze_frame(data_flat_330, f_index_330, i, crop, first_peak)  # order 0.003
        fft_peaks.append(frame)
    end = timeit.default_timer()
    time = end-start
    # # takes around 117 seconds
    # raveled_sa = sa.ravel_scans()
    # for i, frame in enumerate(raveled_sa):
    #     cropped_frame = frame[crop:frame.shape[0] - crop,
    #                     crop:frame.shape[0] - crop]
    #     fft = gen_fft(cropped_frame)
    #     fft_peaks.append(fft[first_peak[0], first_peak[1]])

    plots(cropped_full, np.abs(fft_full_crop))

    plt.show()


if __name__ == "__main__":
    main()

