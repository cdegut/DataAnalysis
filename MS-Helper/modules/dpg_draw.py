from typing import List
import numpy as np
import pandas as pd
from scipy.integrate import quad
from modules.data_structures import MSData
from modules.helpers import multi_bi_gaussian, bi_gaussian
import dearpygui.dearpygui as dpg
from modules.var import colors_list

log_string = ""
def log(message:str) -> None:   
    global log_string
    log_string += message + "\n"
    dpg.set_value("message_box", log_string)

def draw_found_peaks(peaks_centers):
    if dpg.does_item_exist("peak_lines"):
        dpg.delete_item("peak_lines")   
    dpg.add_inf_line_series(peaks_centers, parent="y_axis_plot1", tag="peak_lines")
    dpg.bind_item_theme("peak_lines", "peak_finding_lines")

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

       


    
