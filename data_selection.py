import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pyfftw

from stempy.image import com_dense


def quick_plot(yaxis, color_bar=False, log_norm=False):
    fig, ax = plt.subplots()
    ax.xaxis.tick_top()
    if len(yaxis.shape) == 2:
        if log_norm is False:
            image = ax.imshow(yaxis)
        else:
            image = ax.imshow(yaxis, norm=LogNorm())
        if color_bar is True:
            plt.colorbar(image)
    else:
        plt.plot(yaxis)

    fig.tight_layout()

    # plt.show()


def select_signal(diff_patt, cutoff, verbose = False):
    # num_rows = diff_patt.shape[0]
    # num_cols = diff_patt.shape[1]
    rot_diff_patt = np.transpose(diff_patt)

    row_sum = np.sum(diff_patt, axis=1)
    column_sum = np.sum(rot_diff_patt, axis=1)

    signal_light_rows = np.where(row_sum > cutoff * row_sum.max())[0]
    signal_light_columns = np.where(column_sum > cutoff * column_sum.max())[0]

    center_of_mass = com_dense(diff_patt)
    center_point = np.round(np.flip(np.reshape(center_of_mass, 2))).astype(int)
    # put COM in numpy form and make it into an int
    print(center_point)

    # top_row_cutoff =  signal_light_rows[0]
    # bottom_row_cutoff = signal_light_rows[-1]
    # # bottom_row_cutoff = num_rows - signal_light_rows[-1]
    # left_column_cutoff = signal_light_columns[0]
    # right_column_cutoff = signal_light_columns[-1]
    # right_column_cutoff = num_cols - signal_light_columns[-1]

    row_edges = np.array([signal_light_rows[0], signal_light_rows[-1]])
    col_edges = np.array([signal_light_columns[0], signal_light_columns[-1]])

    center_row_edge_dist = np.average(np.abs(row_edges - center_point[0]))
    center_col_edge_dist = np.average(np.abs(col_edges - center_point[1]))

    # print(center_row_edge_dist)
    # print(center_col_edge_dist)

    radius = round(min(center_row_edge_dist, center_col_edge_dist))
    print(radius)

    top_row_cutoff =  center_point[0] - radius
    bottom_row_cutoff = center_point[0] + radius + 1
    left_column_cutoff = center_point[1] - radius
    right_column_cutoff = center_point[1] + radius + 1

    # cropped_data = diff_patt[]

    # cutoffs = [top_row_cutoff, bottom_row_cutoff, left_column_cutoff, right_column_cutoff]
    # crop_ind = min(cutoffs)

    cropped_data = diff_patt[top_row_cutoff:bottom_row_cutoff, left_column_cutoff: right_column_cutoff]

    # if verbose:
    #     print('Returning a minimum cutoff length of ' + str(crop_ind))
    #     print('The four cutoffs are ' + str(cutoffs))

    return cropped_data, radius


def phase_setup(ravelled_sparse_array):
    """
    Want to produce the location of the Fourier peaks and
    the length between them to use in the phase extraction loop
    without computing it every time in the loop itself
    """
    num_frames = ravelled_sparse_array.shape[0]
    data = ravelled_sparse_array[num_frames // 2]  # get representative frame from center of measurement
    w_frame = pyfftw.empty_aligned((data.shape[0], data.shape[1]), dtype='complex64')
    w_frame[:] = data
    fftw = pyfftw.interfaces.numpy_fft.fft2(w_frame)
    fftw = pyfftw.interfaces.numpy_fft.fftshift(fftw)
    # fft = np.fft.fftshift(np.fft.rfft2(data))
    peaks = fft_find_peaks(fftw, 2)

    zeroth_order = peaks[0, 1:]
    first_order = peaks[1, 1:]

    diff = zeroth_order - first_order
    peak_sep = np.sqrt(np.sum(pow(diff, 2)))
    # The function needs to return an odd value
    # so that the square selection later in algorithm
    # will truly center the Fourier peak inside it
    peak_sep = int(peak_sep * 0.5)
    if peak_sep % 2 == 0:
        peak_sep = peak_sep + 1

    return first_order, peak_sep


def fft_find_peaks(ft, num_peaks):
    """
    Takes fourier transformed image data and returns the coordinates and height of the
    tallest fourier peaks in the dataset. The number of peaks returned is given by
    the argument 'num_peaks'.

    :param ft: fourier transformed 2D image array.
    :type ft: np.ndarray()
    :param num_peaks: The number of fourier peaks we are looking for.
    :type num_peaks: int
    :return: An array of the number of heighest fourier peaks in ft specified by 'num_peaks'
    where each item in the array contains the index and height of a peak in the form of
    [x, y, height]. The array is sorted in descending order by height, so the heighest peak
    is at index o.
    :rtype: np.ndarray()
    """
    ft = np.abs(ft)
    max_vals = np.amax(ft, axis=1)  # loop over ft to find the max values of each row, order E-4
    # find distinct peaks in max value data
    peaks, height = find_peaks(np.abs(max_vals), height=1)  # ~ 4E-5

    # unpack returned values from find_peaks() and store them as array of [height, index] pairs of each peak
    height = height['peak_heights']
    peaks = np.stack((height, peaks), axis=-1)  # E-5

    # sort that array by the first entry in each peak array by descending order
    sorted_peaks = peaks[np.argsort(-peaks[:, 0])]

    # take several of the highest peak values as given by param num_peaks
    max_peaks = sorted_peaks[:num_peaks]

    # reformat so we can loop across our sorted peak row values
    peak_rows = max_peaks[:, 1]
    peak_cols = []

    # this loop finds the column index that corresponds to each value in peak_rows
    for i in peak_rows:  # order E-5
        peak_cols.append(np.argmax(ft[int(i)]))

    peak_cols = np.array(peak_cols)

    # here we join max_peaks with peak_cols so that it holds the data
    # for each peak in the form of [x, y, height]
    max_peaks = np.column_stack((max_peaks, peak_cols))  # order E-6
    max_peaks = max_peaks.astype("u4")
    if len(max_peaks) == 0:
        return np.array([[0, 0, 0]])

    return max_peaks


def grab_square_sect(arr, box_length, center=None):
    # im_dims is a tuple (x, y) of the underlying data shape/dimensions
    # center is a tuple (x, y) of where the center of the mask should be
    # box_length is an int describing how long a side of the final array should be

    if center is None:  # use center of image
        center = (arr.shape[0] // 2, arr.shape[1] // 2)

    halfwidth = int(box_length / 2)

    x_upper_ind = center[0] + halfwidth + 1
    x_lower_ind = center[0] - halfwidth
    y_upper_ind = center[1] + halfwidth + 1
    y_lower_ind = center[1] - halfwidth

    result = arr[x_lower_ind:x_upper_ind, y_lower_ind:y_upper_ind]

    return result


def create_circular_mask(im_dims, radius, center=None):
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


# has to be run prior to raveling, which will restructure the sparse array to be 3D and also sum up all frames
# Trying to calculate np.abs(np.sum(diffraction_pattern)), but, this is the same as just the frame length
def calc_amplitude(sparse_array):
    frames = sparse_array.data
    row_number = sparse_array.shape[0]
    column_number = sparse_array.shape[1]

    amplitudes = np.array([frame.shape[0] for frame in frames])

    ampMap = amplitudes.reshape((row_number, column_number))

    return ampMap

# can make this into a total complex description of a frame (with the phase) by
# t = hx_p * np.exp(1.0j * np.angle(t_temp))

# also needs to run upstream of raveling
def if_empty(sparse_array):
    frames = sparse_array.data
    row_number = sparse_array.shape[0]
    column_number = sparse_array.shape[1]

    lengths = np.array([len(frame) for frame in frames])
    checker = np.where(lengths == 0)[0]
    num_empty_frames = checker.shape

    if num_empty_frames == 0:
        empty = False
    else:
        empty = True
        print(num_empty_frames)

    return empty


def calc_vacuum_kernel(vacuumPath):
    from ncempy.io import dm  # read('data') change

    File = dm.fileDM(vacuumPath)
    data = File.getDataset(0)['data']

    fft = np.fft.fftshift(np.fft.fft2(data))
    kernel = np.conj(fft)

    return kernel


# standard electron holography way of unpacking box around first order to get image of grating
#     ifftw = pe.fftw2D(fourier_space_peak, forward=False)  # inverse Fourier transform area around peak
#     ampMap = ampMap + np.abs(ifftw)

def grab_box(arr, row_num, col_num, center=None):
    """
    Selects a square section with variable length from an array centered at a variable point

    :param arr:  A 2D input array
    :type arr: np.ndarray()
    :param row_num: number of rows in the desired region
    :type row_num: int
    :param col_num: number of columns in the desired region
    :type col_num: int
    :param center: center point of the box to be selected
    :type center: tuple
    :return: square portion around center
    :rtype: np.ndarray()
    """
    if center is None:  # use center of image
        center = (arr.shape[0] // 2, arr.shape[1] // 2)

    half_row = int(row_num / 2)
    half_col = int(col_num / 2)

    row_lower_ind = center[0] - half_row
    row_upper_ind = center[0] + half_row + 1
    col_lower_ind = center[1] - half_col
    col_upper_ind = center[1] + half_col + 1

    result = arr[row_lower_ind:row_upper_ind, col_lower_ind:col_upper_ind]

    return result


def grab_arb_line(data, point1, point2, num, fit_order=3):
    """
    Extract line from point1 to point 2, sampling num points

    :param data: actual array to take data from
    :type data: np.array
    :param point1: first endpoint in pixel coordinates
    :type point1: tuple
    :param point2: second endpoint in pixel coordinates
    :type point2: tuple
    :param num:
    :type num: int
    :param fit_order: exponent of spline fit between points in line
    :type fit_order: int
    :return: line
    """
    x = np.linspace(point1[0], point2[0], num)
    y = np.linspace(point1[1], point2[1], num)

    # Extract the values along the line using cubic interpolation
    from scipy.ndimage import map_coordinates
    line = map_coordinates(data, np.vstack((y, x)), order=fit_order)

    return line


def plane_subtract(arr, center=None, slct_rows=None, slct_columns=None):

    from scipy.optimize import curve_fit

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

    if center is None:
        coef = curve_fit(plane, xdata, arr.ravel())[0]

    if center is not None:
        fit_arr = grab_box(arr, slct_rows, slct_columns, center)
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


def probes_on_image(dm_path, peaks, probe_sep):
    """
    create coordinates to plot a line as long as the probe separation

    use as ax.plot(x, y) after an ax.imshow

    :param dm_path: path to scan's dm3 or dm4 file
    :type dm_path: str
    :param peaks: peaks of the rfft
    :type peaks: np.array
    :param probe_sep: probe separation in nanometers
    :type probe_sep: float
    :return: two arrays of length 2
    :rtype: np.array
    """

    from ncempy.io import read
    from stemh_tools import calc_inplane_angle

    dm = read(dm_path)
    data_shape = dm['data'].shape[0]
    pixel_scale = round(dm['pixelSize'][0], 3)

    length = probe_sep / pixel_scale
    ang = calc_inplane_angle(peaks)

    x = np.linspace(0, np.cos(ang) * length, 2)
    y = np.linspace(0, np.sin(ang) * length, 2)

    if y.sum() < 0:
        y = y + data_shape - 1

    return x, y
