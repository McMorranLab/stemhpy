import numpy as np
import scipy as sci
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy.ndimage import maximum_filter
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pyfftw
#import seaborn as sb #This isn't on talapas
#import imageio
#import os
#import shutil
#I will need to audit the imported packages to see if they actually get used

def apply_window(frame, window_type='hanning'):
    """
    Function that allows for easy window application prior to taking fft
    Arguments:
    frame (1D array) : Data to which window is applied
    window_type (string) : Name of window method to be used

    Returns:
    frame*window (1D array) : array of length(frame) with window applied.
    """
    if window_type=='hanning':
        window = np.hanning(len(frame))
    elif window_type == 'hamming':
        window = np.hamming(len(frame))
    elif window_type == 'blackman':
        window = np.blackman(len(frame))
    elif window_type == 'bartlett':
        window = np.bartlett(len(frame))
    elif window_type == 'kaiser':
        window = np.kaiser(len(frame), beta=14) # beta is a parameter for the Kaiser window
    else:
        raise ValueError("Unsupported window type")
    return frame*window

def build_sample(N, sample_phase, type='step', plot=False):
    if type=='step':
        sample = np.ones(4*N, dtype=np.complex64)
        sample[2*N:3*N]=np.exp(1.0j*sample_phase)
    elif type=='parabolic':
        parabola_array = upside_down_parabola(N,sample_phase)
        sample = np.concatenate((np.ones(2*N, dtype=np.complex64),np.exp(1.0j*parabola_array),np.ones(N, dtype=np.complex64)))
    else:
        print("Phase delay type isn't supported. Code it yourself.")
    

    if plot is True:
        print("You still need to add 1D plotting, you lazy bum.")
        #plt.imshow(np.angle(sample))
        #plt.colorbar()
        #plt.show()

    return sample

def build_sample2D(N, sample_phase, degrees=None, type='step', plot=False):
    if type == 'step':
        sample = np.ones((4*N, 4*N), dtype=np.complex64)
        sample[:, 2*N:3*N] = np.exp(1.0j * sample_phase)
    elif type == 'parabolic':
        parabola_array = upside_down_parabola(N, sample_phase)
        row_sample = np.concatenate((np.ones(2*N, dtype=np.complex64), np.exp(1.0j * parabola_array), np.ones(N, dtype=np.complex64)))
        sample = np.tile(row_sample, (4*N, 1))
    else:
        print("Phase delay type isn't supported. Code it yourself.")

    if degrees is not None:
        sample = sci.ndimage.rotate(sample,degrees,reshape=False,cval=1)

    if plot is True:
        fig,ax = plt.subplots()
        plt.imshow(np.angle(sample))
        plt.xlabel("x position (nm)")
        plt.ylabel("y position (nm)")
        plt.title("Simulated Sample Phase")
        plt.colorbar()
        plt.show()
    
    return sample


def find_fft_peaks(ft, num_peaks=2, filter_strength=25, gaussian_strength=10):
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
        Size of neighborhood a given pixel is compared to when looking for maxima. Should be less than beam separation in pixels
    gaussian_strength: int or float
        standard deviation of 2D gaussian used to smooth list of maxima found after applying filter

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
    smoothed_row_maxima = gaussian_filter(row_maxima,gaussian_strength)

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




def fft_find_peaksd(ft, num_peaks):
    """
    This is different code from Andrew's stemh_analysis function of the same name, although it operates similarly. There is added code to consider
    values along the first row as maximums which is not typical behavior for scipy's find_peaks. I suspect that this might cause some problems once I implement
    pfftw because this code is written for an rfft array. I don't know if pyfftw makes that distinction.
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
    # Ensure ft is an absolute value if it contains complex numbers
    if not np.all(np.isreal(ft)):
        ft = np.abs(ft)

    # Find max values along each row
    max_vals = np.amax(ft, axis=1)

    # Identify peaks using find_peaks()
    peaks, height = find_peaks(max_vals, height=1)
    height = height['peak_heights']

    # Manually check for a peak at the first row (index 0)
    edge_peaks = []
    if max_vals[0] > max_vals[1]:  # Check if first element is a peak
        edge_peaks.append((max_vals[0], 0))

    # Combine detected peaks with manually added edge peak
    peaks = np.stack((height, peaks), axis=-1)
    if edge_peaks:
        peaks = np.vstack((peaks, edge_peaks))

    # Sort peaks by descending height
    sorted_peaks = peaks[np.argsort(-peaks[:, 0])]

    # Select the top num_peaks
    max_peaks = sorted_peaks[:num_peaks]

    # Extract peak row indices
    peaks = max_peaks[:, 0]
    peak_rows = max_peaks[:, 1].astype(int)
    peak_cols = []

    # Find corresponding column indices
    for i in peak_rows:
        peak_cols.append(np.argmax(ft[int(i)]))

    peak_cols = np.array(peak_cols)

    # Format output as [height, row, col]
    max_peaks = np.column_stack((peaks, peak_rows, peak_cols)).astype(int)

    # Handle case where no peaks are found
    if len(max_peaks) == 0:
        return np.array([[0, 0, 0]])

    return max_peaks

def fftw2D(data, forward = True, cores = 4, rfft=True):
    """
    From stemh_tools in main branch. This has not yet been implemented in the main analysis workflow. 
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


def find_2d_peaks(array, num_peaks):
    """
    This completely breaks down when the beams have an area spread as the central peak has local areas with values larger than the values of the other fourier peaks. Please use
    fft_find_peaks
    This is based on the find_2d_peakstwist() function, but is editted to handle junk peaks when
    the beams are not aligned within a row (rfft will still produce duplicates). These tend to be far from the centeral peak and so they are to be filtered by location. Fingers crossed that my shit doesn't get fucked. 
    The work flow is to be:
    Find absolute max first, then filter other peaks so that only peaks that are closer than half of the array size are included.
    Find the coordinates and values of the top 'num_peaks' peaks in a 2D array.
    Each row in the result corresponds to a peak and has the form [y, x, height].
    
    :param array: 2D numpy array to find peaks in.
    :param num_peaks: The number of peaks to return.
    :return: An array of size (num_peaks, 3) where each row is [y, x, height].
    """
    # Flatten the array to work with values and indices
    flat_indices = np.argsort(-array.ravel())  # Sort indices by descending order of values
    sorted_indices = np.unravel_index(flat_indices, array.shape)  # Map flat indices to 2D coordinates
    sorted_coords = np.column_stack((sorted_indices[0], sorted_indices[1]))  # Combine y and x coords

    #Get sorted Values
    sorted_values = array[sorted_coords[:,0],sorted_coords[:,1]]

    #Identify Absolute maximum
    abs_max = sorted_coords[0]
    max_y, max_x = abs_max
    max_peak = sorted_values[0]

    #Cut-off dist is currently hardcoded to be half of the total array size
    y_dist = array.shape[0]//2
    x_dist = array.shape[1]//2

    #Filter local maximum peaks based on x_dist and y_dist
    max_peaks = [(y,x,height)
                for (y,x), height in zip(sorted_coords, sorted_values)
                if abs(y-max_y)<= y_dist and abs(x - max_x)<=x_dist]

    # Extract the values and create the result array [y, x, height]
    #sorted_values = array[sorted_coords[:, 0], sorted_coords[:, 1]]
    #result = np.column_stack((sorted_coords[:num_peaks], sorted_values[:num_peaks]))
    result = np.array(max_peaks[:num_peaks])
    
    return result


def fit_function_1D(measurement, scan_cols, selection_size, bounds, vacuum_amplitudes,first_order_coords,beam_sep_px, phi0_row, phim1_row, trans_fact0, trans_factm1):
    """
    Applies the 1D amplitude and phase fitting method to data passed to it (method assumes a 2D interference pattern). This is fit_function1D with adaptations to handle twisted beams
    
    Args:
    measurement (3D array) : Contains scan position in first dimension, other two dimensions contain interference pattern
    scan_cols (int) : length of first axis of measurement (aka number of interference patterns)
    selection_size (int) : size of region over which to integrate (not fully implemented, value not equal to 1 will throw an error)
    bounds (array of tuples) : Bounds for fit parameters (amp_1, phi_1)
    vacuum_amplitudes (array
    
    
    
    """
    #bounds = [(0, None), (-np.pi, np.pi)]
    phases1D = np.zeros(scan_cols, dtype=np.float32)# 1D array for phases
    transmission_factors1D = np.ones(scan_cols,dtype=np.float32) # I want this to be the same shape as phases.
    for i in range(scan_cols):
        frame = np.copy(measurement[i])
        
        if not frame.any():
            continue  # Skip empty frames

        # Take the Fourier transform of the 2D frame
        ft = np.fft.rfft2(frame)

        #Extract fourier peak
        #fourier_space_peak = grab_sect2D(ft, selection_size, first_order_coord)
        fourier_space_peak = np.sum(grab_area(ft, selection_size, first_order_coords)) #may want to check behavior of np.sum

        # Find locations of the previous beams
        location_0ord = i - beam_sep_px
        location_minus1ord = i - (2 * beam_sep_px)
        
        # Get phases of previous beams (from the 1D array)
        phi_0 = phi0_row[location_0ord] if location_0ord >= 0 else 0
        phi_m1 = phim1_row[location_minus1ord] if location_minus1ord >= 0 else 0

        #Get amplitudes of previous beams
        a_0 = vacuum_amplitudes[1]*trans_fact0[location_0ord] if location_0ord >= 0 else vacuum_amplitudes[1]
        a_m1 = vacuum_amplitudes[2]*trans_factm1[location_minus1ord] if location_0ord >= 0 else vacuum_amplitudes[2]
        
        def objective_function(params, fourier_space_peak):
            amplitude_1, phi_1 = params
            theoretical_intensity = np.conj(a_m1)*a_0*np.exp(1.0j * (phi_0 - phi_m1))+amplitude_1*np.conj(a_0)*np.exp(1.0j * (phi_1 - phi_0))
            # Interference term subtraction
            residual_intensity = fourier_space_peak - theoretical_intensity
            return np.abs(residual_intensity)

        result = minimize(objective_function, [vacuum_amplitudes[0],0], args=(fourier_space_peak),bounds=bounds)

        # Retrieve optimized amplitude and phase
        amplitude_1, phi_1 = result.x
        #Save results
        transmission_factors1D[i] = (amplitude_1/vacuum_amplitudes[0])  # Save all three amplitudes
        phases1D[i] = phi_1  # Save only the leading phase value into phase array
    return phases1D, transmission_factors1D

def fit_vac_amps(I_obs, beam_sep_px, initial_guess = None, showValues = True):
    """I am packing my hard-coded nb cell into a function with the hopes that it will be easier to handle an arbitrary number of beams in the future.
    Fits amplitude values using observed intensity values of peaks.
    args:
    I_obs (1D array of floats) : contains intensity values of each peak from vac frame
    beam_sep_px (int) : distance between peaks in number of array elements
    initial_guess (array of floats) : seed values for the amplitude fitting array
    showValues (bool) : a flag for the optional printing of modeled and observed intensities
    returns:
    optimized_amplitudes (1d array of floats) : optimized amplitude of each probe passing through vacuum
    """
    num_amps = I_obs.size

    #If there is no initial guess passed here, I am going to assume an even distribution.
    # It may be more benefitial to assume a more realistic intensity distribution, but
    # I will worry about that after I get the bulk of the function fleshed out and working.
    if initial_guess == None:
        guess = np.sqrt(I_obs[0]/num_amps)
        initial_guess = np.full (num_amps, guess)

    def normalization_constraint(params):
        return np.sum(np.abs(params)**2) - I_obs[0]#I_tot

    def calculate_intensities(params):
        num_amps = params.size
        I_n = np.zeros(num_amps, dtype=np.float64)
        I_n[0] = np.sum(np.abs(params)**2)
        for i in range (1, num_amps):
            I_n[i] = np.sum(params[:num_amps-i] * params[i:]) 
        return I_n

    def objective_function(params):
        I_model = calculate_intensities(params)
        return np.sum((I_obs - I_model)**2)

    #define bounds and constraints
    bounds = [(0, None)] * num_amps
    constraint = {'type': 'eq', 'fun': normalization_constraint}
    
    # Perform optimization
    result = minimize(objective_function, initial_guess, bounds=bounds,constraints=constraint)

    # Extract optimized amplitudes
    optimized_amplitudes = result.x
    if showValues:
        print("Estimated amplitudes:", optimized_amplitudes)
        I_model = calculate_intensities(optimized_amplitudes)
        print("Modeled intensities:", I_model)
        print("Observed intensities:", I_obs)
        print("Difference in peak intensities: ", I_model - I_obs, "\n\n")
        print('observed total intensity: ', np.sum(I_obs))
        print('Modeled total intensity: ', np.sum(I_model))

    return optimized_amplitudes



def generateCoordinates(xLength,yLength,nx,ny):
    """Generates a mesh grid for the scan coordinates. I think I may have vertical and horizontal switched here, but it shouldn't matter as 
    I am simulating a square case.
    Args:
    xLength (int or float) : Physical span of the x coords (will be centered about zero)
    yLength (int or flota) : Physical span of the y coords (will be centered about zero)
    nx (int) : number of pixels/elements/frames/ in horizontal diraction
    ny (int) : number of pixels/elements/frames/ in vertical diraction

    returns:
    xArray (2D array of floats) : row represents horizontal position, all cols identical
    yArray (2D array of floats) : col represents vertical position, all rows identical
    """
    #xArray, yArray = np.meshgrid(np.linspace(0,xLength,nx),np.linspace(0,yLength,ny)) #d- edit didn't change shit but I've decided it's not a p
    xArray, yArray = np.meshgrid(np.linspace(-xLength//2,xLength//2,nx),np.linspace(-yLength//2,yLength//2,ny))
    return xArray, yArray
    

def grab_area(ft, box_length, center=None):
    """
    Version of grab_sect2D() intended for use when probes are not aligned in one row.
    Args:
    ft (2D array (float or int, idk)): frame containing interference pattern
    box_length (int) : number of pixels decribing length of 2D square array returned
    center (1D array, int) : coordinates of center of selected region
    Returns :
    result (2D array): region surrounding center
    """
    # center is an index (int) of where the center of the selected region should be
    # box_length is an int describing how long the returned array should be
    #arr = ft[0,:]
    center_x = center[1]
    center_y = center[0]

    if center is None:  # use center of image
        #center = arr.size // 2
        center_x = ft.shape[0] // 2
        center_y = ft.shape[1] // 2


    halfwidth = int(box_length / 2)

    x_upper_ind = int(center_x) + halfwidth + 1
    x_lower_ind = (center_x) - halfwidth

    y_upper_ind = int(center_y) + halfwidth + 1
    y_lower_ind = (center_y) - halfwidth

    result = ft[y_lower_ind:y_upper_ind, x_lower_ind:x_upper_ind]

    return result


def simulate_2D_measurement(sample, beam_indices,probe_amps, beam_width, window=True, window_type='hamming'):
    """ 
    I fucked with this function, but I reverted them - normalization sucked and I will just hope that it is handled with pyfftw (install it in talapas folder (does pip work? ubuntu bullshit may be needed), then see pick I took of Kyle's work to show where the computer should look for the library)
    
    Creates data simulation given simulated data and probe profile. This is created from simulate_data2D_intensity() to fix out the source of the weird indexing. 
    Arguments:
    sample (2D array) : complex array representing sample
    beam_indices (list of pairs) : list containing initial beam positions (each pair is a tuple of coordinates)
    probe_amps (1D array) : contains relative amplitudes of probes in order they are listed in beam_indices
    beam_width (int): radius of gausian beam in number of elements
    window (boolean) : flag for if windowing is applied to sample prior to generating data
    window_type (string) : specifies type of windowing used
    Returns:
    measurement (4D array): 3D array containing ft of interference pattern for each probe position
    """
    beam_indices = np.array(beam_indices)
    probe = np.zeros_like(sample, dtype=complex)
    sample_rows, sample_cols = sample.shape
    x_grid, y_grid = np.meshgrid(np.arange(sample_cols),np.arange(sample_rows))
    
    # Loop through the pairs of indices and set corresponding elements to amplitude value
    for (index, (row, col)) in enumerate(beam_indices):
        #probe[row, col] = probe_amps[index]
        amplitude = probe_amps[index]
        # Compute Distrubtion
        dist_squared = (x_grid - col)**2 + (y_grid - row)**2
        gaussian_weight = amplitude * np.exp(-dist_squared/(2*(beam_width)**2))
        probe += gaussian_weight
    #sample_rows, sample_cols = sample.shape
    #scan_rows = sample_rows-beam_indices[-1][0]
    scan_cols = sample_cols - beam_indices[-(len(beam_indices)//2),1]
    scan_rows = sample_rows - beam_indices[-(len(beam_indices)//2),0]

    #Creating data cube
    measurement = np.zeros((scan_rows, scan_cols, sample_rows, sample_cols)) #line w/o +1 term

    for j in np.arange(scan_rows):
        for i in np.arange(scan_cols):# - beam_indices[-(len(beam_indices)//2),1]):
        #for j in np.arange(scan_rows):
            shifted_probe = np.roll(np.roll(probe,shift=i,axis=1),shift=j,axis=0)
            post_sample_wavefunction = shifted_probe * sample
            #Will want to uncomment if I'm to add windowing here
            #if window==True:
                #post_sample_wavefunction = apply_window(post_sample_wavefunction, window_type='hamming')
            # propagate wavefunction to detector
            post_sample_wavefunction_diffraction = np.fft.fft2(post_sample_wavefunction)# / sample.size
            # fftshift to make wavefunction center-aligned like a real measurement
            post_sample_wavefunction_diffraction = np.fft.fftshift(post_sample_wavefunction_diffraction)
            # take picture. Complex -> real :(
            interference_pattern = post_sample_wavefunction_diffraction * np.conjugate(post_sample_wavefunction_diffraction)
            measurement[j,i,:,:] = interference_pattern.real

    return measurement

def simulate_2D_measurement_pointBeam(sample, beam_indices,probe_amps, window=True, window_type='hamming'):
    """ 
    I fucked with this function, but I reverted them - normalization sucked and I will just hope that it is handled with pyfftw (install it in talapas folder (does pip work? ubuntu bullshit may be needed), then see pick I took of Kyle's work to show where the computer should look for the library)
    
    Creates data simulation given simulated data and probe profile. This is created from simulate_data2D_intensity() to fix out the source of the weird indexing. 
    Arguments:
    sample (2D array) : complex array representing sample
    beam_indices (list of pairs) : list containing initial beam positions (each pair is a tuple of coordinates)
    probe_amps (1D array) : contains relative amplitudes of probes in order they are listed in beam_indices
    beam_width (int): radius of gausian beam in number of elements
    window (boolean) : flag for if windowing is applied to sample prior to generating data
    window_type (string) : specifies type of windowing used
    Returns:
    measurement (4D array): 3D array containing ft of interference pattern for each probe position
    """
    beam_indices = np.array(beam_indices)
    probe = np.zeros_like(sample, dtype=complex)
    sample_rows, sample_cols = sample.shape
    #x_grid, y_grid = np.meshgrid(np.arange(sample_cols),np.arange(sample_rows))
    
    # Loop through the pairs of indices and set corresponding elements to amplitude value
    for (index, (row, col)) in enumerate(beam_indices):
        probe[row, col] = probe_amps[index]
    #sample_rows, sample_cols = sample.shape
    #scan_rows = sample_rows-beam_indices[-1][0]
    scan_cols = sample_cols - beam_indices[-(len(beam_indices)//2),1]
    scan_rows = sample_rows - beam_indices[-(len(beam_indices)//2),0]

    #Creating data cube
    measurement = np.zeros((scan_rows, scan_cols, sample_rows, sample_cols)) #line w/o +1 term

    for j in np.arange(scan_rows):
        for i in np.arange(scan_cols):# - beam_indices[-(len(beam_indices)//2),1]):
        #for j in np.arange(scan_rows):
            shifted_probe = np.roll(np.roll(probe,shift=i,axis=1),shift=j,axis=0)
            post_sample_wavefunction = shifted_probe * sample
            #Will want to uncomment if I'm to add windowing here
            #if window==True:
                #post_sample_wavefunction = apply_window(post_sample_wavefunction, window_type='hamming')
            # propagate wavefunction to detector
            post_sample_wavefunction_diffraction = np.fft.fft2(post_sample_wavefunction)# / sample.size
            # fftshift to make wavefunction center-aligned like a real measurement
            post_sample_wavefunction_diffraction = np.fft.fftshift(post_sample_wavefunction_diffraction)
            # take picture. Complex -> real :(
            interference_pattern = post_sample_wavefunction_diffraction * np.conjugate(post_sample_wavefunction_diffraction)
            measurement[j,i,:,:] = interference_pattern.real

    return measurement

def upside_down_parabola(size,height):
    x = np.linspace(-1, 1, size)
    parabola = -0.5 * (x**2) + height
    parabola[parabola < 0] = 0
    return parabola

