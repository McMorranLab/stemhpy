"""
This module contains functions to reconfigure and analyze 4d sparse data in different ways.
"""
from numba import jit, njit, vectorize
import numba
from fft_tools import *


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

    :return: square image array of counted electron interactions
    :rtype: np.ndarray(uint4)
    """
    dp = np.zeros((576 * 576), numba.u4)
    for i in data:
        dp[i] += 1

    dp = dp.reshape(576, 576)

    return dp


@njit()
def elec_count_frame(data, f_index, frame, crop=False):
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

    :param crop: index of distance from the edge of the image array we
    are setting crop distance too. Defaults to False, leaving image
    uncropped
    :type crop: int

    :return: 576 x 576 array of counted electron interactions
    :rtype: np.ndarray(uint4)

    NOTE: There is significant slowdown when large slices of data are specified. If this becomes an
    issue, try to optimize with numba.
        RESOLVED
    I found a weird issue with bounding of param frame: once I specify input over max length, it seems
    to double back on itself...
    I may need to cutoff the input params
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

    if crop:

        return dp[crop: 576 - crop, crop:576 - crop]

    return dp


def analyze_frame(data, f_index, frame, crop, peak):
    """
    Unpacks single sparse data frame into full array, takes the fft
    and returns the value at a fourier peak.

    :param data: flattened 1D array of sparse electron interaction data
    :type data: np.ndarray()

    :param f_index: [n x 2] sized array where n is the number of frames in the data file
    each item in f_index contains the slice indices of one frame of sparse data
    :type f_index: np.ndarray([int, int])

    :param frame: index of the frame we are analyzing in f_index.
    :type frame: int

    :param crop: index of distance from the edge of the image array we
    are setting crop distance too.
    :type crop: int

    :param peak: A two item array that contains the x,y coordinates
    to a fourier peak.
    :type peak: np.ndarray()

    :return: The value of the fourier peak specified by param peak
    :rtype: cint
    """
    counts = elec_count_frame(data, f_index, (frame,), crop) # ~ order of 4E-5
    frame_fft = gen_fft(counts) # ~ order of 0.008, 0.004 for rfft
    fft_peak = frame_fft[peak[0], peak[1]] # ~ order of 9E-7
    return fft_peak


def crop_vacuum(full_data, cutoff, verbose=False):
    """
    Crop out the vacuum space around the edges of
    a full scan

    Adapted from Andrew's algorithm

    :param full_data: 576x576 composite image of electron interactions
    :type full_data: np.ndarray()

    :param cutoff: value between 1 and 0 that provides a threshold of
    where we start the crop
    :type cutoff: float

    :param verbose: Default False. Determines whether or not the program
    returns all of the cutoff indices or just the lowest one.
    :type verbose: bool

    :return: the minimum index for where data is detected, which is the point
    where the crop is set.
    :rtype: int
    """
    rot_diff_patt = np.transpose(full_data)
    # sum each row and column
    row_sum = np.sum(full_data, axis=1)
    column_sum = np.sum(rot_diff_patt, axis=1)
    # look for first values that don't contain vacuum data
    signal_light_rows = np.where(row_sum > cutoff * row_sum.max())[0]
    signal_light_columns = np.where(column_sum > cutoff * column_sum.max())[0]
    
    top_row_cutoff = signal_light_rows[0]
    bottom_row_cutoff = full_data.shape[0] - signal_light_rows[-1]
    left_column_cutoff = signal_light_columns[0]
    right_column_cutoff = full_data.shape[0] - signal_light_columns[-1]

    cutoffs = [top_row_cutoff, bottom_row_cutoff, left_column_cutoff, right_column_cutoff]
    crop_ind = min(cutoffs)

    if verbose:
        return crop_ind, cutoffs
    else:
        return crop_ind

    return None
