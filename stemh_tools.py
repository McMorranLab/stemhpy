import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pyfftw


def fftw2D(data, forward = True, cores = 4, rfft=True):
    """
    Wrapper for FFTW package implementation of a complex 2D Fourier transform
    :param data:  A 2D image array
    :type data: np.ndarray()
    :param forward: The direction of the Fourier transform. True = forward, False = inverse
    :type forward: bool
    :param cores: Number of computer cores the FFT is threaded around
    :type cores: int
    :param rfft: Conditional that tells function whether or not to setup for a real or complex fftw.
    Defauts to True
    :type rfft: Bool
    :return: 2D Fourier transform of given array
    :rtype: np.ndarray()
    """
    # establish bit aligned arrays to prepare the FFT
    if rfft:
        # setup for real fftw scheme
        input = pyfftw.empty_aligned(data.shape, dtype='float32')
        output = pyfftw.empty_aligned((input.shape[0], input.shape[-1] // 2 + 1), dtype='complex64')
    else:
        # setup for a complex fftw scheme
        input = pyfftw.empty_aligned(data.shape, dtype='complex64')
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
    [y, x, height]. The array is sorted in descending order by height, so the highest peak
    is at index 0.
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
    max_peaks = max_peaks.astype("int")
    if len(max_peaks) == 0:
        return np.array([[0, 0, 0]])

    return max_peaks


def grab_square_box(arr, box_length, center = None):
    """
    Selects a square section with variable length from an array centered at a variable point
    :param arr:  A 2D input array
    :type arr: np.ndarray()
    :param box_length: length of side of box to be selected
    :type box_length: int
    :param center: center point of the box to be selected
    :type center: tuple
    :return: square portion around center
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


def calc_box_size(peaks):
    """
    Finds distance between zeroth and first order peaks, then halves this distance.
    When slicing squares around each peak, halving this distance helps keeping
    each peak's surroundings distinct from the other's.
    :param peaks:  A 2D input array of form [y, x, height] from fft_find_peaks
    :type peaks: np.ndarray()
    :return: half the Euclidean distance between the 0th and 1st order peaks in Fourier space
    :rtype: int
    """
    zeroth_order = peaks[0, 1:]
    first_order = peaks[1, 1:]

    diff = first_order - zeroth_order
    peak_sep = np.sqrt(np.sum(pow(diff, 2)))
    half_peak_sep = int(peak_sep // 2)  # use

    return half_peak_sep


def calc_inplane_angle(peaks, deg = False):
    """
    Determines the angle the three diffraction order probes are at in the specimen plane
    :param peaks:  A 2D input array of form [y, x, height] from fft_find_peaks
    :type peaks: np.ndarray()
    :param deg: Switches whether result is given in radians or degrees. Defaults to radians.
    :type deg: bool
    :return: angle between line intersecting all diffraction probes and the +x axis
    :rtype: float
    """
    zeroth_order = peaks[0, 1:]
    first_order = peaks[1, 1:]
    diff = first_order - zeroth_order

    angle = np.arctan2(diff[0], diff[1])

    if deg:
        angle = np.degrees(angle)

    return angle


def calc_amplitude(sparse_array):
    """
    Calculates the STEM signal a given 4D dataset
    Since it has to be used on a 4D dataset, it has to be run upstream of ravelling the sparse array
    Equivalent to computing np.abs(np.sum(diffraction_pattern)) for each frame
    :param sparse_array:  4D sparse dataset
    :type sparse_array: SparseArray()
    :return: STEM signal - an array containing magnitude of each underlying diffraction pattern
    :rtype: np.array()
    """
    frames = sparse_array.data
    row_number = sparse_array.shape[0]
    column_number = sparse_array.shape[1]

    amplitudes = np.array([frame.shape[0] for frame in frames])

    ampMap = amplitudes.reshape((row_number, column_number))

    return ampMap


def kernelize_vacuum_scan(vacuum):
    """
    Takes a vacuum frame (where all three diffracted probes go through vacuum)
    and return the kernel a_0(x)
    :param vacuum: array containing proper vacuum frame
    :type vacuum: np.array()
    :return: complex array containing FFT kernel
    :rtype: np.array()
    """
    fft = np.fft.fftshift(np.fft.rfft2(data))
    kernel = np.conj(fft)

    return kernel
