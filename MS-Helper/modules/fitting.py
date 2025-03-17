from torch import res
from modules.data_structures import MSData, peak_params
import dearpygui.dearpygui as dpg
from modules.helpers import multi_bi_gaussian, bi_gaussian
import numpy as np
import scipy.optimize as opt
from scipy.signal import find_peaks
from modules.dpg_draw import log, draw_fitted_peaks, draw_found_peaks
from scipy.signal import savgol_filter


def peaks_finder_callback(sender, app_data, user_data:MSData):
    spectrum = user_data
    threshold = dpg.get_value("peak_detection_threshold")
    width = dpg.get_value("peak_detection_width")
    distance = dpg.get_value("peak_detection_distance")
    filter_window = dpg.get_value("smoothing_window")
    baseline_window = dpg.get_value("baseline_window")
    sampling_rate = np.mean(np.diff(spectrum.working_data[:,0]))
    max_width = 4*width
    width = width / sampling_rate
    distance = distance / sampling_rate
    peaks_finder(spectrum, threshold, width, max_width, distance, filter_window, baseline_window)

def peaks_finder(spectrum:MSData, threshold:int, width:int, max_width:int, distance:int, filter_window:int, baseline_window:int):
    filtered = spectrum.get_filterd_data(filter_window)
    spectrum.correct_baseline(baseline_window)
    baseline = spectrum.baseline[:,1] 
    filtered_thresolded = np.where(np.abs(filtered - baseline) <= threshold, 0, (filtered - baseline))
    peaks, peaks_data = find_peaks(filtered_thresolded, width=width, distance=distance)
    
    # Delete previous peaks
    peak_to_delete = []
    for old_peak in spectrum.peaks:
        if spectrum.peaks[old_peak].x0_init > spectrum.working_data[:,0][0] and spectrum.peaks[old_peak].x0_init < spectrum.working_data[:,0][-1]:
            peak_to_delete.append(old_peak)
    for peak in peak_to_delete:
        del spectrum.peaks[peak]

    if spectrum.peaks != {}:
        new_peak_index =  max(list(spectrum.peaks.keys())) +1
    else:
        new_peak_index = 0

    # Itterate over the peaks and add them to the dictionary
    i = 0
    for peak in peaks:
        sample = spectrum.working_data[:,0][peak - 250:peak + 250]
        sampling_rate = np.mean(np.diff(sample))
        width = peaks_data["widths"][i] * sampling_rate
        if width > max_width:
            width = max_width
        print(f"Peak {i}: x = {spectrum.working_data[:,0][peak]:.3f}, width = {width:.3f}")
        new_peak = peak_params(A_init=spectrum.working_data[:,1][peak], x0_init=spectrum.working_data[:,0][peak], width=width)
        spectrum.peaks[new_peak_index] = new_peak
        new_peak_index += 1
        i += 1
    
    peaks_centers = spectrum.working_data[:,0][peaks]
    log(f"Detected {len(peaks)} peaks at x = {peaks_centers}")
    
    draw_found_peaks(peaks_centers)

def initial_peaks_parameters(spectrum:MSData):
    initial_params = []
    working_peak_list = []
    i = 0

    if spectrum.peaks is None:
        print("No peaks are detected. Please run peak detection first")
        return
    
    for peak in spectrum.peaks:
        x0_guess = spectrum.peaks[peak].x0_init

        if x0_guess < spectrum.working_data[:,0][0] or x0_guess > spectrum.working_data[:,0][-1]:
            print(f"Peak {i} is out of bounds. Skipping")
            i += 1
            continue

        A_guess = spectrum.peaks[peak].A_init
        sigma_L_guess = sigma_R_guess = spectrum.peaks[peak].width /2
        print(f"Peak {i}: A = {A_guess:.3f}, x0 = {x0_guess:.3f}, sigma_L = {sigma_L_guess:.3f}, sigma_R = {sigma_R_guess:.3f}")
        initial_params.extend([A_guess, x0_guess, sigma_L_guess, sigma_R_guess])
        working_peak_list.append(peak)
        i += 1
    
    if working_peak_list == []:
        print("No peaks are within the data range. Please adjust the peak detection parameters")
        return

    return initial_params, working_peak_list

def rolling_window_fit(sender = None, app_data = None, user_data:MSData = None):
    spectrum = user_data
    baseline_window = dpg.get_value("baseline_window")
    spectrum.correct_baseline(baseline_window)
    initial_params, working_peak_list = initial_peaks_parameters(spectrum)
  
    mbg_param = []
    fitted_peak_list = []
    i = 0
    for peak in working_peak_list:
        window = spectrum.peaks[peak].width
        A_guess, x0_guess, sigma_L_guess, sigma_R_guess =  initial_params[i*4:(i+1)*4]
        initial_param = [A_guess, x0_guess, sigma_L_guess, sigma_R_guess]
        data_x = spectrum.working_data[:,0]
        data_y = spectrum.baseline_corrected[:,1]
        mask = (data_x >= x0_guess - window) & (data_x <= x0_guess + window)
        data_x = data_x[mask]
        data_y = data_y[mask]
        try:
            popt, _ = opt.curve_fit(bi_gaussian, data_x, data_y, p0=initial_param)
            print(f"fitting done for peak {peak}")
            fitted_peak_list.append(peak)
        except:
            print(f"Error - curve_fit failed for peak {peak}")
            spectrum.peaks[peak].fitted = False
            i += 1
            continue

        spectrum.peaks[peak].A_refined = popt[0]
        spectrum.peaks[peak].x0_refined = popt[1]
        spectrum.peaks[peak].sigma_L = popt[2]
        spectrum.peaks[peak].sigma_R = popt[3]
        spectrum.peaks[peak].fitted = True
        mbg_param.extend([popt[0], popt[1], popt[2], popt[3]])
        i += 1
    
    refine_peak_parameters(fitted_peak_list, mbg_param, spectrum)   
    draw_fitted_peaks(None, None, spectrum)

def refine_peak_parameters(working_peak_list, mbg_params, spectrum:MSData):
    #residual = spectrum.baseline_corrected[:,1] - multi_bi_gaussian(spectrum.baseline_corrected[:,0], *mbg_params)
    #dpg.set_value("residual", [spectrum.baseline_corrected[:,0].tolist(), residual.tolist()])
    
    for k in range(50):
        
        i =0
        for peak in working_peak_list:
            x0_fit = spectrum.peaks[peak].x0_refined
            sigma_L_fit = spectrum.peaks[peak].sigma_L
            sigma_R_fit = spectrum.peaks[peak].sigma_R

            # Correct amplitude
            data_x = spectrum.working_data[:,0]
            data_y = spectrum.baseline_corrected[:,1]
            mask = (data_x >= x0_fit - sigma_R_fit) & (data_x <= x0_fit + sigma_L_fit)
            data_x_peak = data_x[mask]
            data_y_peak = data_y[mask]
            #y_filtered = savgol_filter(data_y_peak, window_length=int (sigma_L_fit + sigma_R_fit), polyorder=3)
            peak_residual = np.average(data_y_peak)  - np.average(multi_bi_gaussian(data_x_peak, *mbg_params))
            spectrum.peaks[peak].A_refined = spectrum.peaks[peak].A_refined + (peak_residual/10)
            mbg_params[i*4] = spectrum.peaks[peak].A_refined

            sharpen = False
            if sharpen == True:
                # Sharpen the peak
                L_mask = (data_x >= x0_fit - sigma_L_fit) & (data_x <= x0_fit)
                R_mask = (data_x >= x0_fit) & (data_x <= x0_fit + sigma_R_fit)

                data_x_L = data_x[L_mask]
                data_y_L = data_y[L_mask]
                data_x_R = data_x[R_mask]
                data_y_R = data_y[R_mask]
                residual_L = data_y_L - multi_bi_gaussian(data_x_L , *mbg_params)
                residual_R = data_y_R - multi_bi_gaussian(data_x_R, *mbg_params)
                print(f"Peak {peak}: L = {np.average(residual_L):.3f}, R = {np.average(residual_R):.3f}")
                if np.average(residual_L) > 0:
                    sigma_L_fit = sigma_L_fit * 1.05
                else:
                    pass
                    sigma_L_fit = sigma_L_fit * 0.95    
                
                if np.average(residual_R) > 0:
                    sigma_R_fit = sigma_R_fit * 1.05
                else:
                    pass
                    sigma_R_fit = sigma_R_fit * 0.95

                mbg_params[i*4 + 2] = sigma_L_fit
                mbg_params[i*4 + 3] = sigma_R_fit
                spectrum.peaks[peak].sigma_L = sigma_L_fit
                spectrum.peaks[peak].sigma_R = sigma_R_fit   

            i += 1
            


def update_peak_params(peak_list, popt, spectrum:MSData):
    i = 0
    for peak in peak_list:   
        A_fit, x0_fit, sigma_L_fit, sigma_R_fit = popt[i*4:(i+1)*4]     
        spectrum.peaks[peak].A_refined = A_fit
        spectrum.peaks[peak].x0_refined = x0_fit
        spectrum.peaks[peak].sigma_L = sigma_L_fit
        spectrum.peaks[peak].sigma_R = sigma_R_fit
        spectrum.peaks[peak].fitted = True
        i += 1


'''
def multi_bigaussian_fit(sender = None, app_data = None, user_data:MSData = None):
    spectrum = user_data

    initial_params, working_peak_list = initial_peaks_parameters(spectrum)

    def try_to_fit(data_x:pd.Series, data_y:pd.Series, initial_params, downsample:int = 1):
        try:
            popt, _ = opt.curve_fit(multi_bi_gaussian, data_x[::downsample], data_y[::downsample], p0=initial_params)
            print(f"fitting done with downsampling of {downsample}")
            return popt
        
        except RuntimeError:
            print(f"Error - curve_fit failed with downsampling of {downsample}")
            return None

    print("Peak list:", working_peak_list)

    window = dpg.get_value( "smoothing_window")
    filtered_corrected = savgol_filter(spectrum.baseline_corrected_clipped[:,1], window, 2)
    
    # Initial fitting with downsampled and filtered data
    if len(spectrum.working_data[:,0]) > 25000:
        downsample = int(len(spectrum.working_data[:,0]) /2500)
        popt = try_to_fit(spectrum.working_data[:,0], filtered_corrected, initial_params, downsample)

        if popt is None:
            print("Error - Initial curve_fit failed")
            return
        else:
            update_peak_params(working_peak_list, popt, spectrum)
            draw_fitted_peaks(None, None, spectrum)
    
    else:
        popt = initial_params
    
    # Second fitting with lower downsample
    if len(spectrum.working_data[:,0]) > 10000:
        downsample = 10
        popt = try_to_fit(spectrum.working_data[:,0], filtered_corrected, popt, downsample)
        
        if popt is None:
            return
        else:
            update_peak_params(working_peak_list, popt, spectrum)
            draw_fitted_peaks(None, None, spectrum)
    
    # Final fitting with full resolution data
    popt = try_to_fit(spectrum.baseline_corrected_clipped[:,0], spectrum.baseline_corrected_clipped[:,1], popt)
    
    if popt is None:
        return
    
    update_peak_params(working_peak_list, popt, spectrum)
    draw_fitted_peaks(None, None, spectrum)
    print("Final fitting with full resolution data done")'
'''