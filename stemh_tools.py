import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pyfftw

from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import maximum_filter
from scipy.ndimage import gaussian_filter



def find_fft_peaks(ft, num_peaks=2, filter_strength=25):
    '''
    Finds the locations of peaks in Fourier transforms of interference patterns.
    The phase of these peaks IS the STEMH measurement.
    That is, except for the 0th order (origin) which we want to avoid.

    Parameters
    ----------
    ft: np.ndarray, dtype float or complex
        Fourier transform of interference pattern. Can be the magnitude or complex array.
    num_peaks: int
        Number of determined maxima returned by function.
    filter_strength: int or float
        Size of neighborhood a given pixel is compared to when looking for maxima.

    Returns
    -------
    num_peaks x 2 array of peak locations
    '''
    # conditional to make sure that ft is an absolute value
    if not np.all(np.isreal(ft)):
        ft = np.abs(ft)

    # join all maxima of a Fourier peak together
    filtered_ft = maximum_filter(ft,size=filter_strength)
    # get each row's maximum
    row_maxima = np.max(filtered_ft,axis=1)
    # smooth this discretized signal
    smoothed_row_maxima = gaussian_filter(row_maxima,10)

    # find individual peaks in the smoothed maxima
    # because find_peaks ignores things connected to the end of the array,
    # this should eliminate the 0th order
    peak_rows, peak_heights = find_peaks(smoothed_row_maxima, ft.max()/(10**3))
    peak_heights = peak_heights['peak_heights']

    peak_rows_by_height = peak_rows[np.argsort(peak_heights)][::-1]

    # now need to go from knowing the neighborhood of where the peak is
    # to actually knowing the (x,y) coordinates of the peak
    peak_locs_by_height = np.zeros((peak_rows_by_height.size,2), dtype=int)
    for i, peak_row_num in enumerate(peak_rows_by_height):
        peak_vicinity = np.where(filtered_ft == np.max(filtered_ft[peak_row_num]),
                                ft,0)
        peak_loc = np.unravel_index(np.argmax(peak_vicinity), peak_vicinity.shape)
        peak_locs_by_height[i] = peak_loc

    if num_peaks > peak_locs_by_height.size:
        print("More peaks requested than found.")
        num_peaks = peak_locs_by_height.size
    
    output_peaks = peak_locs_by_height[:num_peaks]

    return output_peaks


def grab_box(arr, square_length=None, center=None, selection_indices=None):
    """
    Selects a section of the input array in one of two ways.
    1. a square of fixed length (square_length) centered around a given point (center), or 
    2. a rectangle with specified corner indices (selection_indices)

    Parameters
    ----------
    arr, np.ndarray
        A 2D input array
    center, 1 x 2 tuple of ints, optional
        center point of box to be selected
    square_length, int, optional
        length of side of box to be selected
    selection_indices, 1 x 4 tuple of ints, optional
        tuple of desired indices set by (first row, last row, first column, last column)

    Returns
    ----------
    2d array of desired region of interest within arr
    """
    if selection_indices is not None and square_length is not None:
        raise Exception("Must use either square_length or selection_indices, not both")

    if square_length is not None:
        if center is None:  # use center of image
            center = (arr.shape[0] // 2, arr.shape[1] // 2)

        halfwidth = int(square_length / 2)

        x_upper_ind = center[0] + halfwidth + 1
        x_lower_ind = center[0] - halfwidth
        y_upper_ind = center[1] + halfwidth + 1
        y_lower_ind = center[1] - halfwidth
    elif selection_indices is not None:
        x_lower_ind, x_upper_ind, y_lower_ind, y_upper_ind = selection_indices

    result = arr[x_lower_ind:x_upper_ind, y_lower_ind:y_upper_ind]

    return result


def plane_subtract(arr, square_length=None, center=None, selection_indices=None):
    """"
    Fit a plane to a region of interest within arr

    If no optional parameters are provided, the whole array will be used for fit

    Parameters
    ----------
    arr, np.ndarray
        2D array with apparent linear slant
    center, 1 x 2 tuple of ints, optional
        center point of box to be selected
    square_length, int, optional
        length of side of box to be selected
    selection_indices, 1 x 4 tuple of ints, optional
        tuple of desired indices set by (first row, last row, first column, last column)

    Returns
    ----------
    2d array subtracted by fit plane broadcasted to whole arr size
    """
    def plane(X, a, b, c):
        x, y = X
        return (a * x + b * y + c)

    row_num = arr.shape[0]
    col_num = arr.shape[1]

    rows = np.arange(row_num, dtype=float)
    cols = np.arange(col_num, dtype=float)

    X, Y = np.meshgrid(rows, cols)
    XX = X.flatten()
    YY = Y.flatten()
    xdata = np.vstack((XX, YY))

    if square_length is None and center is None and selection_indices is None:
        coef = curve_fit(plane, xdata, arr.ravel())[0]

    if center is not None:
        fit_arr = grab_box(arr, 
                           square_length=square_length, 
                           center=center,
                           selection_indices=selection_indices)
        
        frow_num = fit_arr.shape[0]
        fcol_num = fit_arr.shape[1]
        rows = np.arange(frow_num, dtype=float)
        cols = np.arange(fcol_num, dtype=float)

        X, Y = np.meshgrid(rows, cols)
        XX = X.flatten()
        YY = Y.flatten()
        fxdata = np.vstack((XX, YY))

        coef = curve_fit(plane, fxdata, fit_arr.ravel())[0]

    fit_plane = plane(xdata, coef[0], coef[1], coef[2]).reshape(row_num, col_num)
    clean_arr = arr - fit_plane

    return clean_arr


#### depreciated functions
# many are meant to work with fft_find_peaks output [row, column, height]
# which is different from find_fft_peaks output [row, column]

def fft_find_peaks(ft, num_peaks, depreciation_flag=True):
    """
    Identical purpose to, but worse consistency than, find_fft_peaks
    Kept for backwards compatibility

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
    if depreciation_flag:
        print("This function has been superseded by find_fft_peaks, which works better")
        print("To keep using this function without this warning, set depreciation_flag=False")        

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
    Calculates the STEM signal a given 4D Camera dataset
    Since it has to be used on a 4D dataset, it has to be run upstream of unravelling the sparse array
    Equivalent to computing np.abs(np.sum(diffraction_pattern)) for each frame
    :param sparse_array:  4D sparse dataset
    :type sparse_array: stempy.SparseArray()
    :return: STEM signal - an array containing magnitude of each underlying diffraction pattern
    :rtype: np.array()
    """
    frames = sparse_array.data
    row_number = sparse_array.shape[0]
    column_number = sparse_array.shape[1]

    amplitudes = np.array([frame.shape[0] for frame in frames])

    ampMap = amplitudes.reshape((row_number, column_number))

    return ampMap
