from typing import List
from cv2 import threshold
from networkx import sigma
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
import scipy.optimize as opt
from scipy.integrate import quad
from modules.msdata_class import MSData, peak_params
from modules.helpers import multi_bi_gaussian, bi_gaussian
import dearpygui.dearpygui as dpg
from modules.var import colors_list
from time import time

log_string = ""
def log(message:str) -> None:   
    global log_string
    log_string += message + "\n"
    dpg.set_value("message_box", log_string)

def data_clipper(sender = None, app_data = None, user_data:MSData = None):
    spectrum = user_data
    L_clip = dpg.get_value("L_data_clipping")
    R_clip = dpg.get_value("R_data_clipping")
    if L_clip >= R_clip:
        return
    
    spectrum.clip_data(L_clip, R_clip)
    dpg.set_value("original_series", [spectrum.working_data[:,0].tolist(), spectrum.working_data[:,1].tolist()])
    dpg.set_value("corrected_series_plot2", [spectrum.baseline_corrected_clipped[:,0].tolist(), spectrum.baseline_corrected_clipped[:,1].tolist()])
    dpg.set_value("corrected_series_plot3", [spectrum.baseline_corrected_clipped[:,0].tolist(), spectrum.baseline_corrected_clipped[:,1].tolist()])
    filter_data(user_data=spectrum)

    dpg.fit_axis_data("y_axis_plot1")
    dpg.fit_axis_data("x_axis_plot1")
    dpg.fit_axis_data("y_axis_plot2")
    dpg.fit_axis_data("x_axis_plot2")
    dpg.fit_axis_data("y_axis_plot3")
    dpg.fit_axis_data("x_axis_plot3")
    
def filter_data(sender = None, app_data = None, user_data:MSData = None):
    spectrum = user_data
    window_length = dpg.get_value("smoothing_window")
    spectrum.filter_data(window_length)
    dpg.set_value("filtered_series", [spectrum.filtered[:,0].tolist(), spectrum.filtered[:,1].tolist()])

def toggle_baseline(sender = None, app_data = None, user_data:MSData = None):
    spectrum = user_data
    spectrum.baseline_toggle = not spectrum.baseline_toggle
    correct_baseline(None, None, spectrum)

def correct_baseline(sender = None, app_data = None, user_data:MSData = None):
    spectrum = user_data
    if time() - spectrum.last_baseline_corrected < 0.5:
        return
    else:
        window = dpg.get_value("baseline_window")
        spectrum.correct_baseline(window)
        dpg.set_value("baseline", [spectrum.baseline[:,0].tolist(), spectrum.baseline[:,1].tolist()])
        dpg.set_value("corrected_series_plot2", [spectrum.baseline_corrected[:,0].tolist(), spectrum.baseline_corrected[:,1].tolist()])
        dpg.set_axis_limits("y_axis_plot2", min(spectrum.baseline_corrected_clipped[:,1]) - 1, max(spectrum.baseline_corrected_clipped[:,1]))

def peaks_finder(sender = None, app_data = None, user_data:MSData = None):
    spectrum = user_data
    # Assuming y_data and baseline_curve are already defined
    threshold = dpg.get_value("peak_detection_threshold")
    width = dpg.get_value("peak_detection_width")
    distance = dpg.get_value("peak_detection_distance")
    window_length = dpg.get_value("smoothing_window")

    L_clip = dpg.get_value("L_data_clipping")
    R_clip = dpg.get_value("R_data_clipping")

    filtered_clipped = spectrum.filtered[(spectrum.filtered[:,0] > L_clip) & (spectrum.filtered[:,0] < R_clip)]
    
    if len(spectrum.working_data[:,0]) > len(spectrum.baseline):
        correct_baseline(None, None, spectrum)     
    baseline_clipped = spectrum.baseline[(spectrum.baseline[:,0] > L_clip) & (spectrum.baseline[:,0] < R_clip)]
    
    filtered_thresolded = np.where(np.abs(filtered_clipped[:,1] - baseline_clipped[:,1]) <= threshold, 0, (filtered_clipped[:,1] - baseline_clipped[:,1]))
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
        new_peak = peak_params(A=spectrum.working_data[:,1][peak], x0_init=spectrum.working_data[:,0][peak], x0_refined=0, sigma_L=0, sigma_R=0, width=width)
        spectrum.peaks[new_peak_index] = new_peak
        new_peak_index += 1
        i += 1
    
    peaks_centers = spectrum.working_data[:,0][peaks]
    print(f"Detected {len(peaks)} peaks at x = {peaks_centers}")
    log(f"Detected {len(peaks)} peaks at x = {peaks_centers}")

    if dpg.does_item_exist("peak_lines"):
        dpg.delete_item("peak_lines")   
    dpg.add_inf_line_series(peaks_centers, parent="y_axis_plot1", tag="peak_lines")

def initial_peaks_parameters(spectrum):
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

        A_guess = spectrum.working_data[:,0][peak]
        sigma_L_guess = sigma_R_guess = spectrum.peaks[peak].width /2
        print(f"Peak {i}: A = {A_guess:.3f}, x0 = {x0_guess:.3f}, sigma_L = {sigma_L_guess:.3f}, sigma_R = {sigma_R_guess:.3f}")
        initial_params.extend([A_guess, x0_guess, sigma_L_guess, sigma_R_guess])
        working_peak_list.append(peak)
        i += 1
    
    if working_peak_list == []:
        print("No peaks are within the data range. Please adjust the peak detection parameters")
        return
    
    if dpg.get_value("extra_bl"):
        #extra BL peak
        A=max(spectrum.working_data[:,1])*0.1
        x0=spectrum.working_data[:,0][0] + (spectrum.working_data[:,0][-1] - spectrum.working_data[:,0][0]) / 2
        sigma_L = sigma_R = (spectrum.working_data[:,0][-1] - spectrum.working_data[:,0][0]) / 4
        initial_params.extend([A, x0, sigma_L, sigma_R])
        new_peak_index =  min(list(spectrum.peaks.keys())) -1
        working_peak_list.append(new_peak_index)
        print(f"Peak BL: A = {A:.3f}, x0 = {x0:.3f}, sigma_L = {sigma_L:.3f}, sigma_R = {sigma_R:.3f}")


    return initial_params, working_peak_list

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
    print("Final fitting with full resolution data done")

def rolling_window_fit(sender = None, app_data = None, user_data:MSData = None):
    spectrum = user_data
    initial_params, working_peak_list = initial_peaks_parameters(spectrum)
    
    window = dpg.get_value("rolling_window")
    i = 0
    for peak in working_peak_list:
        A_guess, x0_guess, sigma_L_guess, sigma_R_guess =  initial_params[i*4:(i+1)*4]
        initial_param = [A_guess, x0_guess, sigma_L_guess, sigma_R_guess]
        data_x = spectrum.working_data[:,0]
        data_y = spectrum.baseline_corrected_clipped[:,1]
        mask = (data_x >= x0_guess - window) & (data_x <= x0_guess + window)
        data_x = data_x[mask]
        data_y = data_y[mask]
        popt, _ = opt.curve_fit(bi_gaussian, data_x, data_y, p0=initial_param)
        spectrum.peaks[peak].A = popt[0]
        spectrum.peaks[peak].x0_refined = popt[1]
        spectrum.peaks[peak].sigma_L = popt[2]
        spectrum.peaks[peak].sigma_R = popt[3]
        i += 1

    draw_fitted_peaks(None, None, spectrum)

def update_peak_params(peak_list, popt, spectrum:MSData):
    i = 0
    for peak in peak_list:   
        A_fit, x0_fit, sigma_L_fit, sigma_R_fit = popt[i*4:(i+1)*4]     
        if peak < 0:           
            spectrum.peaks[peak] = peak_params(A=A_fit, x0_init=x0_fit, x0_refined=x0_fit, sigma_L=sigma_L_fit, sigma_R=sigma_R_fit, width=sigma_L_fit + sigma_R_fit)
            i += 1
            continue
        spectrum.peaks[peak].A = A_fit
        spectrum.peaks[peak].x0_refined = x0_fit
        spectrum.peaks[peak].sigma_L = sigma_L_fit
        spectrum.peaks[peak].sigma_R = sigma_R_fit
        i += 1

def draw_fitted_peaks(sender = None, app_data = None, user_data:MSData = None):
    spectrum = user_data
    # Delete previous peaks
    for alias in dpg.get_aliases():
        if alias.startswith("fitted_peak_") or alias.startswith("peak_annotation_"):
            dpg.delete_item(alias)

    # Generate fitted curve
    
    peak_list = []
    mbg_param = []

    i = 0
    for peak in spectrum.peaks:
        x0_fit = spectrum.peaks[peak].x0_refined
        if x0_fit < spectrum.working_data[:,0][0] or x0_fit > spectrum.working_data[:,0][-1]:
            continue

        A = spectrum.peaks[peak].A
        sigma_L_fit = spectrum.peaks[peak].sigma_L
        sigma_R_fit = spectrum.peaks[peak].sigma_R
             
        peak_list.append(peak)

        x_individual_fit = np.linspace(x0_fit - 4*sigma_L_fit, x0_fit + 4*sigma_R_fit, 500)
        y_individual_fit = bi_gaussian(x_individual_fit, A, x0_fit, sigma_L_fit, sigma_R_fit)
        mbg_param.extend([A, x0_fit, sigma_L_fit, sigma_R_fit])

        dpg.add_line_series(x_individual_fit, y_individual_fit, label=f"Peak {peak}", parent="y_axis_plot2", tag = f"fitted_peak_{peak}")
        dpg.add_plot_annotation(label=f"Peak {peak}", default_value=(x0_fit, A), offset=(-15, -15), color=[255, 255, 0, 255], clamped=False, parent="gaussian_fit_plot", tag=f"peak_annotation_{peak}")
        i+1
    
    x_fit = np.linspace(np.min(spectrum.working_data[:,0]), np.max(spectrum.working_data[:,0]), 500)
    y_fit = multi_bi_gaussian(x_fit, *mbg_param)   
    if not dpg.does_item_exist("fitted_series"):
        dpg.add_line_series(x_fit, y_fit, label="Fitted Data Series", parent="y_axis_plot2", tag = "fitted_series")
    else:
        dpg.set_value("fitted_series", [x_fit, y_fit])
    
    update_peak_table(spectrum)

def update_peak_table(spectrum:MSData):
    for tag in dpg.get_item_children("peak_table")[1]:
        dpg.delete_item(tag)

    for peak in spectrum.peaks:
        apex = spectrum.peaks[peak].x0_refined
        start = apex - 3 * spectrum.peaks[peak].sigma_L
        end = apex + 3 * spectrum.peaks[peak].sigma_R
        integral = quad(bi_gaussian, start, end, args=(spectrum.peaks[peak].A, spectrum.peaks[peak].x0_refined, spectrum.peaks[peak].sigma_L, spectrum.peaks[peak].sigma_R))[0]
        
        with dpg.table_row(parent = "peak_table"):
            dpg.add_text(f"Peak {peak}")
            dpg.add_text(f"{start:.2f}")
            dpg.add_text(f"{apex:.2f}")
            dpg.add_text(f"{integral:.2f}")


def draw_mz_lines(sender = None, app_data = None, user_data:int = 0):
    k = user_data
    for alias in dpg.get_aliases():
        if alias.startswith(f"peak_annotation_{k}"):
            dpg.delete_item(alias)
    

    mw:int = dpg.get_value(f"molecular_weight_{k}")
    charges:int = dpg.get_value(f"charges_{k}")
    nb_peak_show:int = dpg.get_value(f"nb_peak_show_{k}")
    n = nb_peak_show // 2

    mz_l = []
    z_l = []

    for i in range(-n,n+1):
        z = charges + i
        mz = (mw + z*0.007 ) / z
        mz_l.append(mz)
        z_l.append(z)
        dpg.add_plot_annotation(label=f"{z}+", default_value=(mz, 0), offset=(-15, -15), color=colors_list[k], clamped=False, parent="peak_matching_plot", tag=f"peak_annotation_{k}_{mz}")
        
    if dpg.does_item_exist(f"mz_lines_{k}"):
        dpg.delete_item(f"mz_lines_{k}")
    dpg.add_inf_line_series(mz_l, parent="y_axis_plot3", tag=f"mz_lines_{k}")
    dpg.bind_item_theme(f"mz_lines_{k}", f"mz_line_theme_{k}")

    update_theorical_peak_table(k, mz_l, z_l)


def update_theorical_peak_table(k:int, mz_list:List[float], z_list): 
    dpg.delete_item(f"theorical_peak_table_{k}", children_only=True)
    dpg.delete_item(f"theorical_peak_table_{k}_2", children_only=True)

    i = 0
    with dpg.table_row(parent = f"theorical_peak_table_{k}"):
        for z in mz_list:
            if i <= 5:
                    dpg.add_table_column(label=f"{z_list[i]}+", parent = f"theorical_peak_table_{k}")
                    dpg.add_text(f"{mz_list[i]:.2f}")
            else:
                break
            i += 1

    if i < len(mz_list):
        with dpg.table_row(parent = f"theorical_peak_table_{k}_2"): 
            for r in range(i, len(mz_list)):               
                dpg.add_table_column(label=f"{z_list[i]}+", parent = f"theorical_peak_table_{k}_2")
                dpg.add_text(f"{mz_list[i]:.2f}")
                i += 1

def draw_fitted_peaks_matching(sender = None, app_data = None, user_data:MSData = None):
    spectrum = user_data
    # Delete previous peaks
    for alias in dpg.get_aliases():
        if alias.startswith("fitted_peak_matching") or alias.startswith("peak_annotation_matching_"):
            dpg.delete_item(alias)
    
    threshold = dpg.get_value("peak_matching_threshold")
    start_l = []
    for peak in spectrum.peaks:
        A= spectrum.peaks[peak].A
        apex = spectrum.peaks[peak].x0_refined
        start = apex -  spectrum.peaks[peak].sigma_L
        
        while  bi_gaussian(start, spectrum.peaks[peak].A, apex, spectrum.peaks[peak].sigma_L, spectrum.peaks[peak].sigma_R) > 0.1 * A:       
            start -= 0.01
        start10pcs = start
        while  bi_gaussian(start, spectrum.peaks[peak].A, apex, spectrum.peaks[peak].sigma_L, spectrum.peaks[peak].sigma_R) > 0.01 * A:       
            start -= 0.01
        start1pcs = start
        mid = (start10pcs + start1pcs) / 2
        thick = start10pcs - start1pcs
        dpg.draw_line((mid, 0), (mid, max(spectrum.working_data[:,1])), parent="peak_matching_plot", color=(246, 32, 24,128), thickness=thick, tag=f"fitted_peak_matching_{peak}")
        print(f"Peak {peak}: start1pcs = {start1pcs:.2f}, start10pcs = {start10pcs:.2f}, thick = {thick:.2f}")
        start_l.append(start)
    
   # dpg.add_inf_line_series(start_l, parent="y_axis_plot3", tag="fitted_peak_matching")
   #dpg.bind_item_theme("fitted_peak_matching", "matching_lines")

       


    
