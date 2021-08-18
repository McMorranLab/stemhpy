import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# my attempt to do what Levi did with fft_tools

def quick_plot(yaxis, log_norm = False):

    fig, ax = plt.subplots()
    if len(yaxis.shape) == 2:
        if log_norm is False:
            ax.imshow(yaxis)
        else:
            ax.imshow(yaxis, norm = LogNorm())
    else:
        plt.plot(yaxis)
    plt.show()
    plt.close('all')


def crop_off_vacuum(frms, frm_x, frm_y, cutoff, verbose = False):
    import stempy.image as stim
    diff_patt = stim.calculate_sum_sparse(frms, (frm_y, frm_x))
    rot_diff_patt = np.transpose(diff_patt)

    row_sum = np.sum(diff_patt, axis=1)
    column_sum = np.sum(rot_diff_patt, axis=1)

    signal_light_rows = np.where(row_sum > cutoff * row_sum.max())[0]
    signal_light_columns = np.where(column_sum > cutoff * column_sum.max())[0]

    top_row_cutoff =  signal_light_rows[0]
    bottom_row_cutoff = frm_y - signal_light_rows[-1]
    left_column_cutoff = signal_light_columns[0]
    right_column_cutoff = frm_x - signal_light_columns[-1]

    cutoffs = [top_row_cutoff, bottom_row_cutoff, left_column_cutoff, right_column_cutoff]
    crop_ind = min(cutoffs)

    if verbose:
        return crop_ind, cutoffs
    else:
        return crop_ind


def fft_find_peaks(ft, num_peaks):
    """
    Algorithm/base script written by Levi Brown 210728
    Some relabelling by Andrew Ducharme same date
    Takes fourier transformed image data and returns array of values centered around peak

    return square array

    naive approach, the well of fourier image analysis runs deep...
    """
    max_vals = np.zeros((len(ft)))
    for i in np.arange(len(ft)):
        max_vals[i] = np.max(ft[i])

        # from here we can get the find the best peaks and take some data from around them
    # direct signal processing could be used here, such as scipy.signal.find_peaks() or argrelextrema()

    # find distinct peaks in data
    peaks, height = find_peaks(max_vals, height=1)

    # unpack returned values from find_peaks() and store them as array of [height, index] pairs of each peak
    height = height['peak_heights']
    peaks = np.stack((height, peaks), axis=-1)

    # sort that array by the first entry in each peak array by descending order
    sorted_peaks = peaks[np.argsort(-peaks[:, 0])]
    # take several of the highest peak values as given by param num_peaks
    max_peaks = sorted_peaks[:num_peaks]

    # try to find way of recontextualizing max_peaks in regards to ft
    peak_rows = max_peaks[:, 1]
    peak_cols = []

    for i in peak_rows:
        peak_cols.append(np.argmax(ft[int(i)]))

    peak_cols = np.array(peak_cols)

    peaks_w_loc = np.column_stack((max_peaks, peak_cols))
    # creates 3 x num_peaks array with lateral entries (peak value, col index, row index)

    return peaks_w_loc


def grab_square_sect(arr, box_length, center = None):
    # im_dims is a tuple (x, y) of the underlying data shape/dimensions
    # center is a tuple (x, y) of where the center of the mask should be
    # box_length is an int describing how long a side of the final array should be

    if center is None:  # use center of image
        center = (arr.shape[0] // 2, arr.shape[1] // 2)

    halfwidth = int(box_length / 2)
    x_upper_ind = center[0] + halfwidth
    x_lower_ind = center[0] - halfwidth
    y_upper_ind = center[1] + halfwidth
    y_lower_ind = center[1] - halfwidth

    result = arr[x_lower_ind:x_upper_ind, y_lower_ind:y_upper_ind]

    return result

  
def create_circular_mask(im_dims, radius, center = None):
    # im_dims is a tuple (x, y) of the underlying data shape/dimensions
    # center is a tuple (x, y) of where the center of the mask should be
    # radius is an int describing how large the mask should be

    if center is None:  # use center of image
        center = (im_dims[0]//2, im_dims[1]//2)

    Y, X = np.ogrid[:im_dims[0], :im_dims[1]]
    dist_from_center = np.sqrt(pow(X - center[1], 2) + pow(Y - center[0], 2))

    mask = dist_from_center <= radius
    slice_vecs = np.ogrid[center[0] - radius: center[0] + radius + 1, center[1] - radius: center[1] + radius + 1]

    return mask, slice_vecs
  
