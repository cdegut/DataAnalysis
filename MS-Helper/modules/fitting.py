from modules.data_structures import MSData, peak_params
from typing import Tuple
import dearpygui.dearpygui as dpg
from modules.helpers import multi_bi_gaussian
import numpy as np
from modules.dpg_draw import draw_fitted_peaks
import random

def run_fitting(sender = None, app_data = None, user_data:MSData = None):
    spectrum = user_data
    dpg.show_item("Fitting_indicator")
    rolling_window_fit(spectrum)
    dpg.hide_item("Fitting_indicator")

def initial_peaks_parameters(spectrum:MSData):
    initial_params = []
    working_peak_list = []
    i = 0

    if spectrum.peaks is None:
        print("No peaks are detected. Please run peak detection first")
        return
    
    for peak in spectrum.peaks:
    
        x0_guess = spectrum.peaks[peak].x0_init

        if spectrum.peaks[peak].do_not_fit:
            continue

        if x0_guess < spectrum.working_data[:,0][0] or x0_guess > spectrum.working_data[:,0][-1]:
            print(f"Peak {i} is out of bounds. Skipping")
            i += 1
            continue

        A_guess = spectrum.peaks[peak].A_init
        sigma_L_guess = sigma_R_guess = spectrum.peaks[peak].width /2
        initial_params.extend([A_guess, x0_guess, sigma_L_guess, sigma_R_guess])
        
        working_peak_list.append(peak)
        i += 1
    
    if working_peak_list == []:
        print("No peaks are within the data range. Please adjust the peak detection parameters")
        return

    return initial_params, working_peak_list

def rolling_window_fit(spectrum:MSData):
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
        print(f"Peak {peak}: A = {A_guess:.3f}, x0 = {x0_guess:.3f}, sigma_L = {sigma_L_guess:.3f}, sigma_R = {sigma_R_guess:.3f}")
        
        '''
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
            continue'
        '''
        
        popt = initial_param

        spectrum.peaks[peak].A_refined = popt[0]
        spectrum.peaks[peak].x0_refined = popt[1]
        spectrum.peaks[peak].sigma_L = popt[2]
        spectrum.peaks[peak].sigma_R = popt[3]
        spectrum.peaks[peak].fitted = True
        mbg_param.extend([popt[0], popt[1], popt[2], popt[3]])
        i += 1
    fitted_peak_list = working_peak_list

    refine_peak_parameters(fitted_peak_list, mbg_param, spectrum)   
    draw_fitted_peaks(None, None, spectrum)

def refine_peak_parameters(working_peak_list, mbg_params, spectrum:MSData):

    ###
    # NaN bug seems to come from bad starting parameters for Sigma_L and Sigma_R
    ###
    
    original_peaks: Tuple[int, peak_params] = {peak:spectrum.peaks[peak] for peak in working_peak_list}
    data_x = spectrum.working_data[:,0]
    data_y = spectrum.baseline_corrected[:,1]
    rmse_list = []

    iterations_list = [i for i in range(len(working_peak_list))]
    for k in range(1000):        
        #i =0
        residual = spectrum.baseline_corrected[:,1] - multi_bi_gaussian(spectrum.baseline_corrected[:,0], *mbg_params)
        rmse = np.sqrt(np.mean(residual**2))
        rmse_list.append(rmse)
        if k > 10:
            std = np.std(rmse_list[-10:])
            if std < 0.25:
                dpg.set_value("Fitting_indicator_text","Residual is stable. Done")
                break

            dpg.set_value("Fitting_indicator_text",f"Iteration {k}; RMSE: {rmse:.3f}, Residual std: {std:.3f}")

        random.shuffle(iterations_list) #do not use the same order every iteration

        for i in iterations_list:
            peak = working_peak_list[i]
            x0_fit = spectrum.peaks[peak].x0_refined
            sigma_L_fit = spectrum.peaks[peak].sigma_L
            sigma_R_fit = spectrum.peaks[peak].sigma_R
            if k <= 2:
                print(f"Peak {peak}: x0 = {x0_fit:.3f}, sigma_L = {sigma_L_fit:.3f}, sigma_R = {sigma_R_fit:.3f}")

            mask = (data_x >= x0_fit - sigma_R_fit%4) & (data_x <= x0_fit + sigma_L_fit%4)
            data_x_peak = data_x[mask]
            data_y_peak = data_y[mask]

            try:
                peak_error = np.mean(data_y_peak  - multi_bi_gaussian(data_x_peak, *mbg_params))
            except:
                exit()

            spectrum.peaks[peak].A_refined = spectrum.peaks[peak].A_refined + (peak_error/10)
            mbg_params[i*4] = spectrum.peaks[peak].A_refined

            # Sharpen the peak           
            L_mask = (data_x >= x0_fit - sigma_L_fit) & (data_x <= x0_fit - sigma_L_fit)
            R_mask = (data_x >= x0_fit + sigma_R_fit) & (data_x <= x0_fit + sigma_R_fit)

            data_x_L = data_x[L_mask]
            data_y_L = data_y[L_mask]
            data_x_R = data_x[R_mask]
            data_y_R = data_y[R_mask]
            mbg_L =  multi_bi_gaussian(data_x_L , *mbg_params)
            mbg_R = multi_bi_gaussian(data_x_R, *mbg_params)

            error_l = np.mean((data_y_L - mbg_L))
            error_r = np.mean((data_y_R - mbg_R))

            if error_l > 0 and error_r < 0:
                x0_fit = x0_fit - 0.02

            elif error_l <0 and error_r > 0:
                x0_fit = x0_fit + 0.02
            else:
                sigma_L_fit = sigma_L_fit + error_l/1000
                sigma_R_fit = sigma_R_fit + error_r/1000
            
            original_peak = original_peaks[peak]

            if sigma_L_fit > original_peak.width*4:
                sigma_L_fit = sigma_L_fit/2
            if sigma_R_fit > original_peak.width*4:
                sigma_R_fit = sigma_R_fit/2
            if sigma_L_fit < 1:
                sigma_L_fit = original_peak.width * 1.5
            if sigma_R_fit < 1:
                sigma_R_fit = original_peak.width * 1.5
            

            spectrum.peaks[peak].sigma_L = sigma_L_fit
            spectrum.peaks[peak].sigma_R = sigma_R_fit
            spectrum.peaks[peak].x0_refined = x0_fit

            mbg_params[i*4 + 2] = sigma_L_fit
            mbg_params[i*4 + 3] = sigma_R_fit
            mbg_params[i*4 + 1] = x0_fit

            
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
