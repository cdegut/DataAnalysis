import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse import csc_matrix, diags, spdiags
from scipy.sparse.linalg import spsolve
import scipy.optimize as opt
from scipy.integrate import quad
from modules.msdata_class import MSData, peak_params
from modules.helpers import multi_bi_gaussian, bi_gaussian


spectrum = MSData()
spectrum.import_csv(rf"D:\MassSpec\Um_data.csv")

def data_clipper(sender, app_data):
    L_clip = dpg.get_value("L_data_clipping")
    R_clip = dpg.get_value("R_data_clipping")
    if L_clip >= R_clip:
        return
    spectrum.clip_data(L_clip, R_clip)
    dpg.set_value("original_series", [spectrum.working_data[:,0].tolist(), spectrum.working_data[:,1].tolist()])
    dpg.set_value("corrected_series_plot2", [spectrum.baseline_corrected_clipped[:,0].tolist(), spectrum.baseline_corrected_clipped[:,1].tolist()])
    filter_data()

    if len(spectrum.working_data) > 0:
        dpg.set_axis_limits("x_axis_plot1", L_clip, R_clip)
        dpg.set_axis_limits("y_axis_plot1", min(spectrum.working_data[:,1]) - 0.1, max(spectrum.working_data[:,1]))
        dpg.set_axis_limits("x_axis_plot2", L_clip, R_clip)
        dpg.set_axis_limits("y_axis_plot2", min(spectrum.baseline_corrected_clipped[:,1]) - 0.1, max(spectrum.baseline_corrected_clipped[:,1]))
    
def filter_data(sender = None, app_data = None):
    window_length = dpg.get_value("smoothing_window")
    spectrum.filter_data(window_length)
    dpg.set_value("filtered_series", [spectrum.filtered[:,0].tolist(), spectrum.filtered[:,1].tolist()])

def toggle_baseline(sender = None, app_data = None):
    spectrum.baseline_toggle = not spectrum.baseline_toggle
    correct_baseline()

def correct_baseline(sender = None, app_data = None):    
    window = dpg.get_value("baseline_window")
    spectrum.correct_baseline(window)
    dpg.set_value("baseline", [spectrum.baseline[:,0].tolist(), spectrum.baseline[:,1].tolist()])
    dpg.set_value("corrected_series_plot2", [spectrum.baseline_corrected[:,0].tolist(), spectrum.baseline_corrected[:,1].tolist()])
    dpg.set_axis_limits("y_axis_plot2", min(spectrum.baseline_corrected_clipped[:,1]) - 1, max(spectrum.baseline_corrected_clipped[:,1]))


def peaks_finder():
    # Assuming y_data and baseline_curve are already defined
    threshold = dpg.get_value("peak_detection_threshold")
    width = dpg.get_value("peak_detection_width")
    distance = dpg.get_value("peak_detection_distance")
    window_length = dpg.get_value("smoothing_window")

    L_clip = dpg.get_value("L_data_clipping")
    R_clip = dpg.get_value("R_data_clipping")

    filtered_clipped = spectrum.filtered[(spectrum.filtered[:,0] > L_clip) & (spectrum.filtered[:,0] < R_clip)]
    
    if len(spectrum.working_data[:,0]) > len(spectrum.baseline):
        correct_baseline()     
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

    if dpg.does_item_exist("peak_lines"):
        dpg.delete_item("peak_lines")   
    dpg.add_inf_line_series(peaks_centers, parent="y_axis_plot1", tag="peak_lines")


def multi_bigaussian_fit():

    def try_to_fit(data_x:pd.Series, data_y:pd.Series, initial_params, downsample:int = 1):
        try:
            popt, _ = opt.curve_fit(multi_bi_gaussian, data_x[::downsample], data_y[::downsample], p0=initial_params)
            print(f"fitting done with downsampling of {downsample}")
            return popt
        
        except RuntimeError:
            print(f"Error - curve_fit failed with downsampling of {downsample}")
            return None
    
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
            update_peak_params(working_peak_list, popt)
            draw_fitted_peaks()
    
    else:
        popt = initial_params
    
    # Second fitting with lower downsample
    if len(spectrum.working_data[:,0]) > 10000:
        downsample = 10
        popt = try_to_fit(spectrum.working_data[:,0], filtered_corrected, popt, downsample)
        
        if popt is None:
            return
        else:
            update_peak_params(working_peak_list, popt)
            draw_fitted_peaks()
    
    # Final fitting with full resolution data
    popt = try_to_fit(spectrum.baseline_corrected_clipped[:,0], spectrum.baseline_corrected_clipped[:,1], popt)
    
    if popt is None:
        return
    
    update_peak_params(working_peak_list, popt)
    draw_fitted_peaks()
    print("Final fitting with full resolution data done")

def update_peak_params(peak_list, popt):
    i = 0
    for peak in peak_list:
        A_fit, x0_fit, sigma_L_fit, sigma_R_fit = popt[i*4:(i+1)*4]
        spectrum.peaks[peak].A = A_fit
        spectrum.peaks[peak].x0_refined = x0_fit
        spectrum.peaks[peak].sigma_L = sigma_L_fit
        spectrum.peaks[peak].sigma_R = sigma_R_fit
        i += 1

def draw_fitted_peaks():
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
    
    update_peak_table()

def update_peak_table():
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



    

# Create a Dear PyGui context
dpg.create_context()


with dpg.window(label="Control", width=1450, height= 10, no_close=True, no_collapse=True, no_move=True, no_resize=True):
    # Add a slider to adjust the window length
    with dpg.group(horizontal=True, horizontal_spacing= 50):
        dpg.add_text("Data Clipping:")
        min_value = min(spectrum.original_data[:,0])
        max_value = max(spectrum.original_data[:,0])
        dpg.add_slider_int(label="Data clipping left", width=400, default_value=min_value, min_value=min_value, max_value=max_value, tag="L_data_clipping", callback=data_clipper)
        dpg.add_slider_int(label="Data clipping right", width=400, default_value=max_value, min_value=min_value, max_value=max_value, tag="R_data_clipping", callback=data_clipper)
    
# Create a window
with dpg.window(label="Data Filtering and peak finding",width=1430, pos=(0,100), no_close=True, no_move=True, no_resize=True, tag="Data Filtering"):
    
    # Create a plot for the data
    with dpg.plot(label="Data Filtering", width=1430, height=600, tag="data_plot") as plot1:
        # Add x and y axes
        x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="m/z", tag= "x_axis_plot1")
        y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Y Axis", tag = "y_axis_plot1")
        
        dpg.add_line_series(spectrum.working_data[:,0].tolist(), spectrum.working_data[:,1].tolist(), label="Original Data Series", parent=y_axis, tag="original_series")
        dpg.add_line_series(spectrum.filtered[:,0].tolist(), spectrum.filtered[:,1].tolist(), label="Filtered Data Series", parent=y_axis, tag="filtered_series")
        dpg.add_line_series(spectrum.baseline[:,0].tolist(), spectrum.baseline[:,1].tolist(), label="Snip Baseline", parent=y_axis, tag="baseline")  
        
    dpg.add_text("Data Filtering:")
    dpg.add_slider_int(label="Smoothing Window", default_value=100, min_value=3, max_value=1000, callback=filter_data, tag="smoothing_window")
    dpg.add_text("Baseline estimation:")
    with dpg.group(horizontal=True, horizontal_spacing= 50):
        dpg.add_button(label="Toggle Baseline", callback=toggle_baseline)
        dpg.add_slider_int(label="Baseline window", default_value=500, min_value=100, max_value=1000, tag="baseline_window")
        dpg.add_button(label="Update Baseline", callback=correct_baseline)
      
    dpg.add_text("Peak detection:")
    dpg.add_slider_int(label="Peak detection distance from baseline", default_value=100, min_value=1, max_value=300, tag="peak_detection_threshold")
    dpg.add_slider_int(label="Peak width", default_value=200, min_value=1, max_value=600, tag="peak_detection_width")
    dpg.add_slider_int(label="Peak min distance", default_value=1000, min_value=1, max_value=1000, tag="peak_detection_distance")
    dpg.add_button(label="Find Peaks", callback=peaks_finder)


with dpg.window(label="Peak fitting", width=1430, height=900, pos=(0,120), no_close=True, no_move=True, no_resize=True):
    # Create a plot for the raw data
    with dpg.plot(label="Gaussian Fit", width=1430, height=600, tag="gaussian_fit_plot") as plot2:
        # Add x and y axes
        dpg.add_plot_axis(dpg.mvXAxis, label="m/z", tag="x_axis_plot2")
        dpg.add_plot_axis(dpg.mvYAxis, label="Y Axis", tag="y_axis_plot2")
        
        # Add the raw data series to the plot
        dpg.add_line_series(spectrum.baseline_corrected_clipped[:,0], spectrum.baseline_corrected_clipped[:,1], label="Corrected Data Series", parent="y_axis_plot2", tag="corrected_series_plot2")
    
    with dpg.group(horizontal=True, horizontal_spacing= 50):
        dpg.add_button(label="Multi Fit Gaussians", callback=multi_bigaussian_fit)
        dpg.add_button(label="Redraw Peaks", callback=draw_fitted_peaks)
        
    with dpg.table(header_row=True, tag="peak_table"):
        dpg.add_table_column(label="Peak Label")
        dpg.add_table_column(label="Peak Start")
        dpg.add_table_column(label="Peak Apex")
        dpg.add_table_column(label="Peak Integral")

# Import the custom theme
from modules.dpg_style import general_theme, data_theme
dpg.bind_theme(general_theme)
dpg.bind_item_theme("original_series", data_theme)
dpg.bind_item_theme("corrected_series_plot2", data_theme)

# Create a viewport and show the plot
dpg.create_viewport(title='Multi Bi Gaussian Fit', width=1450, height=1000)
dpg.focus_item("Data Filtering")
dpg.show_style_editor()
dpg.show_metrics()
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()