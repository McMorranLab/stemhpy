# base python modules
import glob
import time
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import cpu_count
from skimage.restoration import unwrap_phase

# less standard modules
import pyfftw
import py4DSTEM

# our module
import stemh_tools as st


## HUMAN-INPUT
# paths hard coded from your own device
dataPath = '/Users/andrewducharme/University of Oregon Dropbox/UO-McMorran Lab/ParticipantFiles/Ducharme/Data/STEMH/2306_NCEM_Magnetics/Data/23-06-07/ef_4dstem_edge.dm4'
baseSavePath = '/Users/andrewducharme/University of Oregon Dropbox/UO-McMorran Lab/ParticipantFiles/Ducharme/Data/STEMH/2306_NCEM_Magnetics/Data/23-06-07/'
timing = False
verbose = False

for path in glob.glob(dataPath + '*.dm4'):
    fileName = st.get_file_name(path)
    if verbose:
        print(fileName)
    savePath = baseSavePath + fileName + '/'
    st.mkdir(savePath)

    datacube = py4DSTEM.import_file(path)
    data = datacube.data

    datacube, hf_mask = datacube.filter_hot_pixels(thresh=0.05, return_mask=True)

    scan_row_num = datacube.shape[0]  
    scan_col_num = datacube.shape[1]  
    frame_row_num = datacube.shape[2]
    frame_col_num = datacube.shape[3]

    # create virtual STEM image of data

    vSTEM = np.sum(datacube.data, axis=(2,3))

    fig, ax = plt.subplots(tight_layout=True)
    im = ax.imshow(vSTEM)
    plt.colorbar(im)
    fig.savefig(savePath + 'vSTEM.png')

    # Find the location of the peaks in the Fourier transform of an interference pattern. 
    # The peak location is essentially constant throughout the scan.

    frame_location = np.unravel_index(np.argmax(vSTEM),vSTEM.shape)
    # using frame with most intensity (i.e. most likely to be vacuum)
    if verbose:
        print(frame_location)

    test_frame = data[frame_location]

    # find the first order index by computing the real fft to match what we use in the loop
    test_rfft = np.fft.rfft2(test_frame)

    first_order, second_order = st.find_fft_peaks(test_rfft)  # find 2 highest magnitude peaks in vac_rfft
    if verbose:
        print(first_order)
        print(second_order)
    orders=(first_order,second_order)

    # setting up pyfftw numpy interface
    pyfftw.config.NUM_THREADS = cpu_count()
    pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'
    pyfftw.interfaces.cache.enable()

    base = np.empty((frame_row_num, frame_col_num), dtype='float32')
        
    # data array made 3D, from (scan_row, scan_col, : ,:) to (scan_row * scan_col, :,:)
    num_frames = scan_col_num * scan_row_num
    data1D = data.reshape((num_frames, frame_row_num, frame_col_num))

    for i, order in enumerate(orders):
        order_num = i + 1
        for selection_size in [1,3,5,7]:
            # initialize arrays to store values through loop
            peaks = np.zeros(num_frames, dtype=complex)
        
            if timing:
                start = time.time()

            # the forward Fourier transform is the vast majority of the work+computation time here
            for i, frame in enumerate(data1D):
                base[:] = frame

                ft = pyfftw.interfaces.numpy_fft.rfft2(base)  # take Fourier transform of the windowed frame
                # ft = cupy.fft.rfft2(cupy.asarray(frame)) # uncomment for gpus

                fourier_space_peak = st.grab_box(ft, selection_size, order) # select the area around desired peak

                peaks[i] = np.sum(fourier_space_peak)

            phaseMap = np.angle(peaks).reshape((scan_row_num, scan_col_num))

            if timing:
                end = time.time()
                print("Total time (s): " + str(end - start))

            # save raw reconstruction
            savePathDetails = 'phase_ord' + str(order_num) + '_ss' + str(selection_size)
            fig, ax = plt.subplots(tight_layout=True)
            ax.imshow(phaseMap)        
            fig.savefig(savePath + savePathDetails + '.png', dpi = 150)
            np.save(savePath + savePathDetails, phaseMap)

            # save unwrapped and rough plane-subtracted reconstruction
            savePathDetails = 'phase_unwrapped_ord' + str(order_num) + '_ss' + str(selection_size)
            fig, ax = plt.subplots(tight_layout=True)
            ax.imshow(st.plane_subtract(unwrap_phase(phaseMap)))        
            fig.savefig(savePath + savePathDetails + '.png', dpi = 150)