from typing import List, Tuple
from anyio import key
from matplotlib.pylab import f
from networkx import sigma
import numpy as np
from scipy.integrate import quad
from sympy import is_zero_dimensional
from modules import render_callback
from modules.data_structures import MSData
from modules.helpers import multi_bi_gaussian, bi_gaussian
import dearpygui.dearpygui as dpg
from modules.render_callback import RenderCallback
from modules.var import colors_list

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
    dpg.set_value("baseline", [spectrum.baseline[:,0].tolist(), spectrum.baseline[:,1].tolist()])
    dpg.set_value("corrected_series_plot2", [spectrum.baseline_corrected[:,0].tolist(), spectrum.baseline_corrected[:,1].tolist()])
    dpg.set_value("corrected_series_plot3", [spectrum.baseline_corrected[:,0].tolist(), spectrum.baseline_corrected[:,1].tolist()])
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
    filtered = spectrum.get_filterd_data(window_length)
    dpg.set_value("filtered_series", [spectrum.working_data[:,0].tolist(),  filtered])

def toggle_baseline(sender = None, app_data = None, user_data:MSData = None):
    spectrum = user_data
    spectrum.baseline_toggle = not spectrum.baseline_toggle
    spectrum.request_baseline_update()

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
        if not spectrum.peaks[peak].fitted:
            continue

        A = spectrum.peaks[peak].A_refined
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
        if not spectrum.peaks[peak].fitted:
            continue
        apex = spectrum.peaks[peak].x0_refined
        start = apex - 3 * spectrum.peaks[peak].sigma_L
        end = apex + 3 * spectrum.peaks[peak].sigma_R
        integral = quad(bi_gaussian, start, end, args=(spectrum.peaks[peak].A_refined, spectrum.peaks[peak].x0_refined, spectrum.peaks[peak].sigma_L, spectrum.peaks[peak].sigma_R))[0]
        
        with dpg.table_row(parent = "peak_table"):
            dpg.add_text(f"Peak {peak}")
            dpg.add_text(f"{start:.2f}")
            dpg.add_text(f"{apex:.2f}")
            dpg.add_text(f"{integral:.2f}")


    
            
        
                    
                

        
