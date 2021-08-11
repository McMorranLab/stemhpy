"""
This module is just an example of how to use the fft_tools.py and frame_analysis.py modules
"""
import h5py
import stempy.image as stim
import stempy.io as stio
from stempy.io.sparse_array import SparseArray
from fft_tools import *


def main():

    sparse_330 = SparseArray.from_hdf5('data/data_scan330_th4.0_electrons_centered.h5')

    data_330 = sparse_330.data
    f_index_330 = frame_index(data_330)
    data_flat_330 = np.concatenate(data_330).astype('u4')

    frames_330 = elec_count_frame(data_flat_330, f_index_330, (0, 10000))
    fft_frames_330 = gen_fft(frames_330)

    peaks = fft_find_peaks(np.abs(fft_frames_330),5)
    mask_330 = fft_mask(fft_frames_330, peaks[2], 70, 'square')

    plots(frames_330, np.abs(fft_frames_330))
    plots(frames_330 * mask_330, np.abs(fft_frames_330) * mask_330)

    plt.show()


if __name__ == "__main__":
    main()
