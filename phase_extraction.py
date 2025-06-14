"""
Author: Andrew Ducharme

From GitHub
"""
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pyfftw
from ncempy.io import dm


def fftw2D(data, forward = True, cores = 4):
    """
    Wrapper for FFTW package implementation of a complex 2D Fourier transform
    :param data:  A 2D image array
    :type data: np.ndarray()
    :param forward: The direction of the Fourier transform. True = forward, False = inverse
    :type forward: bool
    :param cores: Number of computer cores the FFT is threaded around
    :type cores: int
    :return: 2D Fourier transform of given array
    :rtype: np.ndarray()
    """
    input = pyfftw.empty_aligned(data.shape, dtype='complex64')  # establish bit aligned arrays to prepare the FFT
    output = pyfftw.empty_aligned(data.shape, dtype='complex64')
    input[:] = data  # assign the data to the input array

    # execute Fourier transform
    if forward:
        fftw = pyfftw.FFTW(input, output, axes=(0, 1), threads=cores)
        fft = pyfftw.interfaces.numpy_fft.fftshift(fftw())  # take 0 frequency component to the center
    else:
        fftw = pyfftw.FFTW(input, output, axes=(0, 1), threads=cores, direction = 'FFTW_BACKWARD')
        fft = fftw()

    return fft


def fft_find_peaks(ft, num_peaks):
    """
    Takes Fourier transformed image data and returns the coordinates and height of the
    highest Fourier peaks in the dataset. The number of peaks returned is given by
    the argument 'num_peaks'.
    :param ft:  A Fourier transformed 2D image array.
    :type ft: np.ndarray()
    :param num_peaks: The number of fourier peaks we are looking for.
    :type num_peaks: int
    :return: An array of the number of highest Fourier peaks in ft specified by 'num_peaks'
    where each item in the array contains the index and height of a peak in the form of
    [x, y, height]. The array is sorted in descending order by height, so the highest peak
    is at index o.
    :rtype: np.ndarray()
    """
    # conditional to make sure that ft is an absolute value
    if not np.all(np.isreal(ft)):
        ft = np.abs(ft)

    # loop over ft to find the max values of each row
    max_vals = np.amax(ft, axis=1) # order E-4
    # find distinct peaks in max value data
    peaks, height = find_peaks(max_vals, height=1)  # ~ 4E-5

    # unpack returned values from find_peaks() and store them as array of [height, index] pairs of each peak
    height = height['peak_heights']
    peaks = np.stack((height, peaks), axis=-1)  # E-5

    # sort that array by the first entry in each peak array by descending order
    sorted_peaks = peaks[np.argsort(-peaks[:, 0])]

    # take several of the heighest peak values as given by param num_peaks
    max_peaks = sorted_peaks[:num_peaks]

    # reformat so we can loop across our sorted peak row values
    peaks = max_peaks[:, 0]
    peak_rows = max_peaks[:, 1]
    peak_cols = []

    # this loop finds the column index that corresponds to each value in peak_rows
    for i in peak_rows:  # order E-5
        peak_cols.append(np.argmax(ft[int(i)]))

    peak_cols = np.array(peak_cols)

    # here we join max_peaks with peak_cols so that it holds the data
    # for each peak in the form of [height, row, col] ==> [height, y, x]
    max_peaks = np.column_stack((peaks, peak_rows, peak_cols))  # order E-6
    max_peaks = max_peaks.astype("u4")
    if len(max_peaks) == 0:
        return np.array([[0,0,0]])

    return max_peaks


def grab_square_sect(arr, box_length, center = None):
    """
    Wrapper for FFTW package implementation of a complex 2D Fourier transform
    :param arr:  A 2D input array
    :type data: np.ndarray()
    :rtype: np.ndarray()
    """

    if center is None:  # use center of image
        center = (arr.shape[0] // 2, arr.shape[1] // 2)

    halfwidth = int(box_length / 2)

    x_upper_ind = center[0] + halfwidth + 1
    x_lower_ind = center[0] - halfwidth
    y_upper_ind = center[1] + halfwidth + 1
    y_lower_ind = center[1] - halfwidth

    result = arr[x_lower_ind:x_upper_ind, y_lower_ind:y_upper_ind]

    return result


def phase_setup(ravelled_sparse_array, crop = False, rfft = False):
    """
    Want to produce the location of the Fourier peaks and
    the length between them to use in the phase extraction loop
    without computing it every time in the loop itself
    """
    num_frames = ravelled_sparse_array.shape[0]
    data = ravelled_sparse_array[num_frames // 2]  # get representative frame from center of measurement

    if crop:

        data = data[crop: 576 - crop, crop:576 - crop]

    if rfft:

        fft = np.fft.fftshift(np.fft.rfft2(data))

    else:

        fft = fftw2D(data)

    peaks = fft_find_peaks(fft, 2)

    zeroth_order = peaks[0, 1:]
    first_order = peaks[1, 1:]

    diff = zeroth_order - first_order
    peak_sep = np.sqrt(np.sum(pow(diff, 2)))
    half_peak_sep = int(peak_sep // 2)

    return first_order, half_peak_sep


def calc_vacuum_kernel(vacuumPath, crop=False):
    """
    Takes the vacuum scan from a .dm4 file and return the sharply peaked kernel a_0(x)
    :param vacuumPath: string of the path to the location of the dm4 file in question
    :type ft: str
    :return: complex array containing FFT kernel
    :rtype: np.ndarray()
    """
    File = dm.fileDM(vacuumPath)
    data = File.getDataset(0)['data']
    if crop:

        data = data[crop: 576 - crop, crop:576 - crop]

    fft = np.fft.fftshift(np.fft.fft2(data))
    kernel = np.conj(fft)

    return kernel