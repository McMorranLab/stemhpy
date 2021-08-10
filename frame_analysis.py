"""
I want to find the fft_peak of each frame and store that as an image dataset

    do some math with vacuum data and sample data to try and get phase

I could also compare that against a low-resolution analysis with blocks of frames

brute force it with for loops, or try a vectorize approach (either thru numba or np)

try further optimizations with sparse array based algorithms

like I said, brute force it then do some optimization
"""
import numpy as np
from numba import jit, njit, vectorize
import numba
from fft_tools import *


def frame_index(data):
    """Get lengths of each sub array in data, lots of room for improvement"""
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

    :return: 576 x 576 array of counted electron interactions
    :rtype: np.ndarray(uint4)
    """
    dp = np.zeros((576 * 576), numba.u4)
    for i in data:
        dp[i] += 1

    dp = dp.reshape(576, 576)

    return dp


@njit()
def elec_count_frame(data, f_index, frame):
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

    return dp
