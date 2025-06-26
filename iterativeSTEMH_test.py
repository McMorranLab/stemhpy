
#Standard Packages
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
from scipy.optimize import minimize
#import pyfftw Not yet implemented in this version
import py4DSTEM

#Our package
import realistic_stemh as spr

#User Inputs
#file_path = '/gpfs/projects/mcmorran_lab/dball/testData.npy'
file_path = '/gpfs/projects/mcmorran_lab/dball/iterativeStemh/cube_mag640kx_stepSize5ang_CL1p3m_.npy'
#save_path = '/gpfs/projects/mcmorran_lab/dball/iterativeStemh/analysis_output/'
save_path = '/gpfs/home/dball/'

verbose = True
graphDiffSpots = False


# load data from file and filter hotpixels
#datacube = py4DSTEM.import_file(file_path)
data = np.load(file_path)#datacube.data
#datacube, hf_mask = datacube.filter_hot_pixels(thresh=0.5,return_mask=True)

#scan_row_num = datacube.shape[0]  
#scan_col_num = datacube.shape[1]  
#frame_row_num = datacube.shape[2]
#frame_col_num = datacube.shape[3]

scan_row_num = data.shape[0]  
scan_col_num = data.shape[1]  
frame_row_num = data.shape[2]
frame_col_num = data.shape[3]

# Reconstruction Initialization
## This is run using og cell. May need to exchange set-up for the second cell that uses find_2d_peaks instead of fft_find_peaks
num_peaks = 3
vac_frame = np.abs(np.fft.rfft2(data[1,1,:,:]))
rfft_peaks = spr.find_fft_peaks(vac_frame, num_peaks, filter_strength=5,gaussian_strength=1)
#rfft_peaks = spr.fft_find_peaks(vac_frame, num_peaks) #This the original function and identifies incorrect 2nd ord peak when using real data

  

#zeroth_order = rfft_peaks[0,1:].astype(int) #preAndrew's Alg
#first_order = rfft_peaks[1,1:].astype(int)
#second_order = rfft_peaks[2,1:].astype(int)
zeroth_order = rfft_peaks[0].astype(int)
first_order = rfft_peaks[1].astype(int)
#second_order = rfft_peaks[2].astype(int)

beam_sep_px =  int(np.linalg.norm(first_order - zeroth_order))#This is used in intensity-modelling for amplitude fitting cell, may be redundant with the vector. I will eventually decide what to keep.

beam_sep_vec = first_order - zeroth_order
if verbose:
    print("rfft peaks: ",rfft_peaks)
    print("beam_sep_px: ",beam_sep_px)
    print('zeroth_order: ',zeroth_order)
    print('first_order: ',first_order)
    #print('second_order: ',second_order)
    print("beam seperation vection: ",beam_sep_vec)


#Creating and initializing observed intensity array
num_probes = rfft_peaks.shape[0]
I_obs = np.zeros(num_probes, dtype=np.float64)
for i in range(num_probes):
    row, col = rfft_peaks[i]
    I_obs[i] = vac_frame[row,col]


#Initializing intensity values for amplitude-informed contruction
#Andrew's code doesn't return the intensity :( I don't think I need this block with the changes I've made to the vac amp fitting
"""zeroth_intensity = vac_frame[zeroth_order]
first_intensity = vac_frame[first_order]
second_intensity = vac_frame[second_order]
if verbose:
    print("rfft peaks: ",rfft_peaks)
    print('zeroth_intensity: ',zeroth_intensity)"""


if graphDiffSpots:
    #Raw vac frame
    fig,ax = plt.subplots()
    plt.title("Vacuum Frame Data")
    plt.imshow(data[1,1,:,:])
    plt.colorbar()
    plt.savefig(save_path + "vacuumDataFrame")

    #Vac frame rfft diffraction
    fig,ax = plt.subplots()
    plt.title("Vacuum Frame Image")
    plt.imshow(vac_frame)
    plt.colorbar()
    plt.savefig(save_path + "vacuumFrame_rfft")
    
    #Below plots serve as a sanity check, but are not generally needed
    #Vac frame rfft diffraction log
    fig,ax = plt.subplots()
    plt.title("Log Scale Vacuum Frame Image")
    plt.imshow(np.log(vac_frame))
    plt.colorbar()
    plt.savefig(save_path + "vacuumFrame_rfft_log")
    
    #Vac frame rfft diffraction spots
    fig,ax = plt.subplots()
    plt.title("Vacuum Frame Image 0th Peak")
    plt.imshow(vac_frame[0:100,0:100])
    plt.colorbar()
    plt.savefig(save_path + "vacuumFrame_rfft_0peak")

    fig,ax = plt.subplots()
    plt.title("Vacuum Frame Image 1st Peak")
    plt.imshow(vac_frame[211:311,174:274])
    plt.colorbar()
    plt.savefig(save_path + "vacuumFrame_rfft_1peak")


#####################################################
#####################################################
########  Phase Iteration wo Amp Fitting  ###########
#####################################################
#####################################################
#scan_rows = datacube.shape[0]  
#scan_cols = datacube.shape[1]
scan_rows = data.shape[0]  
scan_cols = data.shape[1]
vert_beam_offset = beam_sep_vec[0]
hor_beam_offset = beam_sep_vec[1]
"""
# Loop over the desired selection sizes; here we use selection_size == 1
for selection_size in [1,3, 5, 7]:  # Could also be [1, 3, 5, 7], etc.
    # Initialize the phases array.
    # Its shape will be (num_rows, num_cols) corresponding to the first two axes of measurement.
    phases = np.zeros((scan_rows, scan_cols), dtype=np.float32)

    # Iterate over the first two axes of the 4D array.
    for i in range(scan_cols):
        for k in range(scan_rows): # k is value running over right edge
            # Extract the 2D frame corresponding to position (k, i)
            frame = data[k, i]
            
            # If the frame is entirely zero, set the phase at this position to 0 and continue.
            if not frame.any():
                phases[k, i] = 0
                #continue

            # Take the Fourier transform of the 2D frame.
            ft = np.fft.rfft2(frame) #* (frame.size/2)
            
            # Extract the Fourier-space region around the first peak.
            fourier_space_peak = np.sum(spr.grab_area(ft, selection_size, first_order))
            #print(fourier_space_peak.shape)
            # Determine the indices for previous beam orders along the first axis.
            # (Assuming that the beam separation is only along the first axis for each fixed k.)
            location_col_1ord = i
            location_col_0ord = i - hor_beam_offset
            location_col_minus1ord = i - 2 * hor_beam_offset

            location_row_1ord = k
            location_row_0ord = k - vert_beam_offset
            location_row_minus1ord = k - 2 * vert_beam_offset

            
            # Get the phase from previous beams; if out-of-bounds, assume vacuum (phase = 0).
            if location_row_0ord < 0:
                phi_0 = 0
            elif location_col_0ord <0:
                phi_0 = 0
            else:
                phi_0 = phases[location_row_0ord,location_col_0ord]

            if location_row_minus1ord < 0:
                phi_m1 = 0
            elif location_col_minus1ord <0:
                phi_m1 = 0
            else:
                phi_m1 = phases[location_row_minus1ord,location_col_minus1ord]

            
            
            # Compute the interference from the previous orders.
            zero_neg1_interference = np.exp(1.0j * (phi_0 - phi_m1))
            
            # Subtract the interference to isolate the first-order beam.
            first_isolation = fourier_space_peak - zero_neg1_interference
            
            # Compute the recovered phase.
            # Use np.angle to extract the phase and then add the phase of the previous beam.
            # It is assumed that recovered_phase is an array-like object so take its first element.
            recovered_phase = np.angle(first_isolation) + phi_0
            #print(type(recovered_phase))
            # Store the recovered phase in the phases array.
            # (If recovered_phase is a scalar, remove the `[0]` indexing.)
            phases[k,i] = recovered_phase



    fig,ax = plt.subplots()
    plt.title(f"Phase Corrected Reconstruction\n(Selection Size: {selection_size})")
    #plt.imshow(np.angle(sample))
    #plt.imshow(-phases,extent=[X.min()+8, X.max(), Y.min(), Y.max()])
    plt.imshow(-phases)
    plt.colorbar()
    plt.savefig(save_path + f"BasicPhaseIter_reconstruction_ss{selection_size}")
    #plt.show()
"""

#####################################################
#####################################################
########  Phase Iteration w/ Amp Fitting  ###########
#####################################################
#####################################################

#Vacuum Amplitude Estimation
"""I_obs = np.array([zeroth_intensity, first_intensity,second_intensity])   # observed intensities
n_orders = np.arange(len(I_obs))   # Match the number of rfft output points

I_tot = np.sum(I_obs)

#N = measurement.shape[1]
spacing_index = beam_sep_px  # Spacing between probes in units of the FFT grid
#delta_phi = 2 * np.pi * spacing_index / N  # Phase shift remains the same

def normalization_constraint(params):
    return np.sum(np.abs(params)**2) - zeroth_intensity#I_tot

# Model function to calculate theoretical intensities
def calculate_intensities(params):
    a1, a0, an1 = params  # Amplitudes of the probes
    I_n = np.array([np.abs(a1)**2+np.abs(a0)**2+np.abs(an1)**2,an1*a0+a0*a1, an1*a1])
    return I_n

# Negative log-likelihood function
def negative_log_likelihood(params):
    I_model = calculate_intensities(params)
    return np.sum((I_obs - I_model)**2)

# Initial guesses for amplitudes
initial_guess = [.3, .4, .4]

constraint = {'type':'eq', 'fun':normalization_constraint}

# Perform optimization
result = minimize(negative_log_likelihood, initial_guess, bounds=[(0, None), (0, None), (0, None)],constraints=constraint)

# Extract optimized amplitudes
optimized_amplitudes = result.x


# Validate modeled intensities
I_model = calculate_intensities(optimized_amplitudes)

if verbose:
    print("Estimated amplitudes:", optimized_amplitudes)
    print("Modeled intensities:", I_model)
    print("Observed intensities:", I_obs)
    print('observed total intensity: ', I_tot)
    print('Modeled total intensity: ', np.sum(I_model))"""
opt_amps = spr.fit_vac_amps(I_obs, beam_sep_px)
optimized_amplitudes = np.array([opt_amps[1], opt_amps[0],opt_amps[1]])

#scan_rows = datacube.shape[0]  
#scan_cols = datacube.shape[1]
scan_rows = data.shape[0]  
scan_cols = data.shape[1]

# Initialize the phases array
phases = np.zeros((scan_rows, scan_cols), dtype=np.float32)
transmission_factor = np.ones((scan_rows, scan_cols), dtype=np.float32)
#print(amplitudes.shape)
vert_beam_offset = beam_sep_vec[0]
hor_beam_offset = beam_sep_vec[1]


# Loop over the desired selection sizes
for selection_size in [1,3]:  # Can also be [1, 3, 5, 7], etc.
    bounds = [(0, optimized_amplitudes[0]), (-np.pi, np.pi)]
    
    for k in range(scan_rows):
        #print("k value (row) for current computation is ", k)
        vert_location_0ord = k - vert_beam_offset
        vert_location_m1ord = k - 2*vert_beam_offset
        if (vert_location_0ord)<0:
            #non-probe beams are set to vacuum values. May want this as optional arg instead but idk how to add that in function rn
            phi_0_row = np.zeros(scan_cols)
            transmission_amp_0 = np.ones_like(phi_0_row)
            phi_m1_row = np.zeros_like(phi_0_row)
            transmission_amp_m1 = np.ones_like(phi_0_row)
            #Ord 0 amd ord 1 have vacuum filler
        elif (k - 2*vert_beam_offset)<0:
            phi_0_row = np.copy(phases[k-vert_beam_offset,:])
            transmission_amp_0 = np.copy(transmission_factor[k-vert_beam_offset,:])
            phi_m1_row = np.zeros(scan_cols)
            transmission_amp_m1 = np.ones(scan_cols)
            #Minus 1 ord has filler, but 0 ord is predetermined
        else:
            phi_0_row = np.copy(phases[k-vert_beam_offset,:])
            transmission_amp_0 = np.copy(transmission_factor[k-vert_beam_offset,:])
            phi_m1_row = np.copy(phases[k-2*vert_beam_offset,:])
            transmission_amp_m1 = np.copy(transmission_factor[k-vert_beam_offset,:])
            #create the arrays. . . 
            """This is where the 5/5/25 adaptation work ended. I need to consider what I am passing to fit_function1(). The size of arguments passed in array form will grow with addition
            probes contributing to the reconstruction. This method fixes the value persistence present when the reconstruction occurs for the entire image. Do scans always raster over a 
            square image? This method can still be parallelized so it might not be the worst solution. . . """

        #change fit_function1D to accept the arrays. . . 
        
        measurement_row = data[k]
        #phase_row, factor_row = fit_function_1Dtest(measurement_row, scan_cols, selection_size, bounds, optimized_amplitudes,first_order[1],hor_beam_offset, phi_0_row, phi_m1_row,transmission_amp_0,transmission_amp_m1)
        phase_row, factor_row = spr.fit_function_1D(measurement_row, scan_cols, selection_size, bounds, optimized_amplitudes,first_order,hor_beam_offset, phi_0_row, phi_m1_row,transmission_amp_0,transmission_amp_m1)
        phases[k,:] = np.copy(phase_row)
        transmission_factor[k,:] = np.copy(factor_row)

            

    fig, ax = plt.subplots()
    plt.title(f"Phase Mapping with Amplitude Consideration\n(selection size: {selection_size})")
    plt.imshow(-phases)
    plt.colorbar()
    plt.savefig(save_path + f"AmpPhaseIter_reconstruction_ss{selection_size}")
    #plt.show()

    fig, ax = plt.subplots()
    plt.title(f"Transmission Factor Mapping\n(selection size: {selection_size})")
    plt.imshow(transmission_factor)
    plt.colorbar()
    plt.savefig(save_path + f"transmissionFactorMap_ss{selection_size}")
    #plt.show()

#print(transmission_factor.shape)