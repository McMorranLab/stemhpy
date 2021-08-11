"""
This module contains functions that analyze the fft of 4D sparse data sets
"""
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks
from frame_analysis import *


def gen_fft(data):
    """
    This functions generates a 2d fft from a square frame image.

    :param data: A square image array that holds frame data such as electron counts over
    a slice of frames.
    :type data: np.ndarray()

    :return: A shifted 2d fft of the image data
    :rtype: np.ndarray()
    """
    return np.fft.fftshift(np.fft.fft2(data))


def plots(data, fft_data):
    """
    Simple plotting function: plots the frame data image and fft image next to each other
    NOTE: fft_data must be passed as its absolute value
    """
    fg, ax = plt.subplots(1, 2)

    ax[0].imshow(data)
    ax[1].imshow(fft_data, norm=LogNorm())


def fft_find_peaks(ft, num_peaks):
    """
    Takes fourier transformed image data and returns the coordinates and height of the
    heighest foureir peaks in the dataset. The number of peaks returned is given by
    the argument 'num_peaks'.

    :param ft: The absolute value of a fourier transformed 2D image array.
    :type ft: np.ndarray()

    :param num_peaks: The number of fourier peaks we are looking for.
    :type num_peaks: int

    :return: An array of the number of heighest fourier peaks in ft specified by 'num_peaks'
    where each item in the array contains the index and height of a peak in the form of
    [x, y, height]. The array is sorted in descending order by height, so the heighest peak
    is at index o.
    :rtype: np.ndarray()
    """
    max_vals = np.zeros((len(ft)))
    # loop over ft to find the max values of each row
    for i in np.arange(len(ft)):
        max_vals[i] = np.max(ft[i])

        # find distinct peaks in max value data
    peaks, height = find_peaks(max_vals, height=1)

    # unpack returned values from find_peaks() and store them as array of [height, index] pairs of each peak
    height = height['peak_heights']
    peaks = np.stack((height, peaks), axis=-1)

    # sort that array by the first entry in each peak array by descending order
    sorted_peaks = peaks[np.argsort(-peaks[:, 0])]

    # take several of the heighest peak values as given by param num_peaks
    max_peaks = sorted_peaks[:num_peaks]

    # reformat so we can loop across our sorted peak row values
    peak_rows = max_peaks[:, 1]
    peak_cols = []

    # this loop finds the column index that corresponds to each value in peak_rows
    for i in peak_rows:
        peak_cols.append(np.argmax(ft[int(i)]))

    peak_cols = np.array(peak_cols)

    # here we join max_peaks with peak_cols so that it holds the data
    # for each peak in the form of [height, x, y]
    max_peaks = np.column_stack((max_peaks, peak_cols))'
    max_peaks[:, [1, 2]] = max_peaks[:, [2, 1]]

    return max_peaks


def fft_mask(ft, peak, radius, shape):
    """
    Creates a boolean mask of an input fft array centered on a specified peak.
    The shape and size of the mask are specified by args 'shape' and 'radius' respectively.

    :param ft: Absolute value of a fourier transformed sample image.
    :type ft: np.ndarray()

    :param peak: An array of peak coordinates stored as [x, y, height].
    :type peak: np.ndarray()

    :param radius: Radius of mask array in the case of a circular mask, or the max distance
    in x and y values from the origin in the case of a square mask.
    :type radius: int

    :param shape: The shape of the mask, only accepts 'circle' or 'square', anything else
    will raise a ValueError.
    :type shape: str

    :return: A boolean mask where excluded values are 0 and kept values are 1.
    :rtype: np.ndarray()
    """
    # create meshgrid that physically stores dimensions of ft as a grid in index
    # space. Each point on the grid corresponds to a value of x and y that gives its index
    xx, yy = np.meshgrid(np.arange(0, ft.shape[0]), np.arange(0, ft.shape[0]))

    # circular mask
    if shape == 'circle':

        # shift the index grid so (0, 0) lands on our 'peak' argument
        mx = xx - int(peak[0])
        my = yy - int(peak[1])

        # calculate the distance from the origin of each point on the grid,
        # points that fall within our specified 'radius' are stored as a 1
        # and points outside that radius are stored as 0
        mask = np.sqrt(mx ** 2 + my ** 2)
        bool_mask = np.where(mask <= radius, 1, 0)

    # square mask
    elif shape == 'square':

        # shift the index grid so (0, 0) lands on our 'peak' argument
        # and take the absolute value
        mx = np.abs(xx - int(peak[0]))
        my = np.abs(yy - int(peak[1]))

        # find all points that fall within our specified radius along the x and y axes
        # using our 1 and 0 for points that fall within and outside of that range respectively
        # multiply the two together to get the square mask
        # D = {(mx,my)|-radius <= mx <= radius, -radius <= my <= radius}
        bool_x = np.where(mx <= radius, 1, 0)
        bool_y = np.where(my <= radius, 1, 0)

        bool_mask = bool_x * bool_y

    # when unrecognized mask shape is given
    else:

        raise ValueError("Mask shape not recognized. Allowed mask shapes: 'circle' and 'square'.")

    return bool_mask
