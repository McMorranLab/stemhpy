"""
This module contains functions to reconfigure and analyze 4d sparse data in different ways.
"""
import timeit

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit, vectorize, guvectorize, prange
import numba
from fft_tools import *
import phase_extraction as pe
from scipy.stats import norm
from sklearn.cluster import KMeans, DBSCAN


def frame_index(data):
    """
    Takes input 4d sparse array and returns an array of slices that correspond to the dimensions of
    the sub-arrays within the flattened 4d sparse array

    :param data: 4d sparse data array
    :type data: np.ndarray()

    :return: A 1D array of 1D arrays that contain the slice indices of the sub-arrays within the sparse data. The
    indices are stored in the format (i_min, i_max)
    :rtype: np.ndarray()
    """
    f_index = np.zeros((len(data), 2), "u4")
    position = 0

    for i in np.arange(len(data)):
        f_index[i, 0] = position
        f_index[i, 1] = position + len(data[i])
        position += len(data[i])

    return f_index


@njit()
def elec_count_full(data):
    """
    Counts each electron interaction from a flattened 4D sparse array
    Compiled with numba.njit()

    :param data: 1D array of unsigned 32-bin ints that reperesent electron interaction index
    :type data: np.ndarray(uint4)

    :return: 576 x 576 image array of counted electron interactions
    :rtype: np.ndarray(uint4)
    """
    dp = np.zeros((576 * 576), numba.u4)
    for i in data:
        dp[i] += 1

    dp = dp.reshape(576, 576)

    return dp


@njit()
def elec_count_frame(data, f_index, frame, crop):
    """
    Counts electron interactions within specified range of frames
    from a flattened 4D sparse array

    :param data: 1D array of unsigned 32-bin ints that reperesent electron interaction index
    :type data: np.ndarray(uint4)

    :param f_index: [n x 2] sized array where n is the number of frames in the data file
    each item in f_index contains the slice indices of one frame of sparse data
    :type f_index: np.ndarray([int, int])

    :param frame: index or slice indeces of desired frame(s) of data
    :type frame: int or tuple(int, int)

    :param crop: The indices at which to crop the final output array of
    electron count data. if crop = np.array([0, 576, 0, 576]) the frame will
    remain uncropped.
    :type crop: np.ndarray()

    :return: square array of counted electron interactions
    :rtype: np.ndarray(uint4)
    """
    dp = np.zeros((576 * 576), numba.u4)

    # slice of frames
    if len(frame) == 2:
        slice_min = f_index[frame[0], 0]
        slice_max = f_index[frame[1], 1]

    # single frame
    else:
        slice_min = f_index[frame[0], 0]
        slice_max = f_index[frame[0], 1]

    for i in data[slice_min: slice_max]:
        dp[i] += 1

    dp = dp.reshape(576, 576)

    return dp[crop[0]:crop[1], crop[2]:crop[3]]


def analyze_frame(data, f_index, frame, crop, mask_size, peak, kernel_peak):
    """
    Unpacks single sparse data frame into full array, takes the fft
    and returns the value at a fourier peak. If the data is cropped,
    this algorithm will also find the new peak location by scaling the
    original peak location by the magnitude of the crop

    :param data: flattened 1D array of sparse electron interaction data
    :type data: np.ndarray()

    :param f_index: [n x 2] sized array where n is the number of frames in the data file
    each item in f_index contains the slice indices of one frame of sparse data
    :type f_index: np.ndarray([int, int])

    :param frame: index of the frame we are analyzing in f_index.
    :type frame: int

    :param crop: The indices to which the data is cropped to. Returns a
    square array
    :type crop: np.ndarray()

    :param mask_size: The width of the box of data we grab that is centerd
    on the peak location
    :type mask_size: int

    :param peak: A two item array that contains the coordinates
    to a fourier peak.
    :type peak: np.ndarray()

    :param kernel_peak: box of vacuum data to use in phase computation
    :type kernel_peak: np.ndarray()

    :return: A list containing the amplitude and phase of the box centered
    on the 1st order fourier peak in data
    :rtype: List
    """
    if type(crop) == bool:
        # when the frame is uncropped
        crop = np.array([0, 576, 0, 576])
        counts = elec_count_frame(data, f_index, (frame,), crop)
        frame_fft = gen_fft(counts)
        fourier_space_peak = pe.grab_square_sect(frame_fft, mask_size, peak)

    else:
        counts = elec_count_frame(data, f_index, (frame,), crop)
        frame_fft = gen_fft(counts)
        crop_shape = np.array(frame_fft.shape)

        # find the new location of the peak in the cropped frame
        # using scaling
        r_row = (peak[0] - 288) / 576
        r_col = (peak[1] - 144) / 289   # crop
        # r_col = (peak[1] - 288)/ 576
        peak_row = round((crop_shape[0] * r_row) + crop_shape[0] / 2)
        peak_col = round((crop_shape[1] * r_col) + crop_shape[1] / 2)
        shifted_peak = np.array([peak_row, peak_col])

        fourier_space_peak = pe.grab_square_sect(frame_fft, mask_size, shifted_peak)

    # amplitude computation
    ifftw = np.fft.ifft2(fourier_space_peak)
    ampMap = np.abs(ifftw)

    # phase computation
    t_temp = np.sum(kernel_peak * fourier_space_peak)  # convolve kernel and first order peak (* multiplies elementwise)
    phase = np.angle(t_temp)

    return [ampMap, phase]


@njit()
def crop_frame(data, x_center, y_center):
    """
    Finds crop indices of an input array. The algorithm finds anchor points at the
    intersection of the boundary of the data array and the axes on which the
    center of mass of the data lie. From there it finds the nearest data point
    to each of those anchor points and takes there indices as the location to
    apply the crop. Compiled with numba

    :param data: sparse array containing the [row, column] locations of electron
    interactions on a 576 x 576 pixel array.
    :type data: np.ndarray()

    :param x_center: the mean value of the distribution of points along the
    x-axis
    :type x_center: float

    :param y_center: the mean value of the distribution of points along the
    y-axis
    :type y_center: float

    :return: an array containing the crop distance from each boundary stored
    as [distance from right, distance from left, distance from top, distance from bottom]
    :rtype: np.ndarray()
    """

    max_dist = np.sqrt(2 * 576 ** 2)
    r_crop = [max_dist, 0, 0]   # right crop index
    l_crop = [max_dist, 0, 0]   # left crop index
    t_crop = [max_dist, 0, 0]   # top crop index
    b_crop = [max_dist, 0, 0]   # bottom crop index

    for point in data:
        r_dist = np.sqrt((point[0] - y_center) ** 2 + point[1] ** 2)
        l_dist = np.sqrt((point[0] - y_center) ** 2 + (point[1] - 576) ** 2)
        t_dist = np.sqrt(point[0] ** 2 + (point[1] - x_center) ** 2)
        b_dist = np.sqrt((point[0] - 576) ** 2 + (point[1] - x_center) ** 2)

        if r_dist < r_crop[0]:
            r_crop = [r_dist, point[1], point[0]]

        if l_dist < l_crop[0]:
            l_crop = [l_dist, point[1], point[0]]

        if t_dist < t_crop[0]:
            t_crop = [t_dist, point[1], point[0]]

        if b_dist < b_crop[0]:
            b_crop = [b_dist, point[1], point[0]]

    return np.array([r_crop[1], 576 - l_crop[1], t_crop[2], 576 - b_crop[2]])


def get_dense_fast(sparse):
    """
    Quick algorithm to extract the row and column indices of an array of
    sparse values. Assumes that the sparse data comes from a 576 x 576 array.

    :param sparse: A sparse array containing electron interactions on a
    576 x 576 pixel array.
    :type sparse: np.ndarray()

    :return: A sparse array containing the index location [row, column] on
    a 576 x 576 array for each item in sparse
    :rtype: np.ndarray()
    """
    y = sparse % 576  # rows
    x = sparse // 576  # cols
    return np.stack((y, x), axis=1)


def crop_vacuum(sparse_frame):
    """
    Takes in a sparse array of electron interactions and computes the location
    of each item, finds the crop distance from each side, selects the smallest
    distance, and creates a square crop centered on the distribution that contains
    the maximum possible data while excluding vacuum data.

    :param sparse_frame: sparse array containing electron interaction on a
    576 x 576 pixel array
    :type sparse_frame: np.ndarray()

    :return: the slice indices for a square crop of a 576 x 576 dense array
    :rtype: np.ndarray()
    """
    # Todo: fix the bounds of the crop algorithm
    # I got a more legible pattern when I changed the center
    if len(sparse_frame) == 0:
        return np.array([0, 576, 0, 576])

    dense_frame = get_dense_fast(sparse_frame)
    crop_vals = crop_frame(dense_frame, 288, 288).astype(int)
    max_dist = np.array([288 - crop_vals[0], 288 - crop_vals[1],
                         288 - crop_vals[2], 288 - crop_vals[3]])
    max_dist = max_dist.max()
    sq_crop_vals = np.array([288 - max_dist, 288 + max_dist,
                             288 - max_dist, 288 + max_dist])
    return sq_crop_vals


def cluster_crop(frame):
    """

    :param frame:
    :return:
    """
    if len(frame) == 0:
        return 0

    frame = frame.astype(int)
    f_stack = get_dense_fast(frame)
    DBcluster = DBSCAN(eps=20, min_samples=8).fit_predict(f_stack) + 1
    cluster = DBcluster * frame
    cluster = cluster[cluster != 0]
    cluster = get_dense_fast(cluster)
    crop = crop_frame(cluster)
    return crop

# # here's the timer code:
# from timeit import Timer
# t = Timer("", globals=globals())
# time = t.timeit(1000) / 1000
