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


def draw_mz_lines(sender = None, app_data = None, user_data:Tuple[MSData,int] = None):
    render_callback: RenderCallback = user_data[0]
    k = user_data[1]
    
    mw:int = dpg.get_value(f"molecular_weight_{k}")
    charges:int = dpg.get_value(f"charges_{k}")
    nb_peak_show:int = dpg.get_value(f"nb_peak_show_{k}")

    mz_l = []
    z_l = []
    z_mz = []

    for i in range(nb_peak_show):
        z = charges - i
        mz = (mw + z*0.007 ) / z
        mz_l.append(mz)
        z_l.append(z)
        z_mz.append((z, mz))


    for alias in dpg.get_aliases():
        if alias.startswith(f"mz_lines_{k}"):
            dpg.delete_item(alias)

    for line in mz_l:
        center_slice = render_callback.spectrum.baseline_corrected[:,1][(render_callback.spectrum.baseline_corrected[:,0] > line - 0.5) & (render_callback.spectrum.baseline_corrected[:,0] < line + 0.5)]
        max_y = max(center_slice) if len(center_slice) > 0 else 0
        dpg.draw_line((line, -50), (line, max_y), parent="peak_matching_plot", tag=f"mz_lines_{k}_{line}", color=colors_list[k], thickness=1)

    update_theorical_peak_table(k, mz_l, z_l)
    render_callback.mz_lines[k] = z_mz
    redraw_blocks(render_callback)

def update_theorical_peak_table(k:int, mz_list:List[float], z_list): 
    dpg.delete_item(f"theorical_peak_table_{k}", children_only=True)

    dpg.add_table_column(parent = f"theorical_peak_table_{k}")
    dpg.add_table_column(parent = f"theorical_peak_table_{k}")
    dpg.add_table_column(parent = f"theorical_peak_table_{k}")


    row = len(mz_list)//3 + (len(mz_list)%3 >0)
    for r in range(0, row ):
        with dpg.table_row(parent = f"theorical_peak_table_{k}", tag = f"theorical_peak_table_{k}_{r}"):
            dpg.bind_item_theme( f"theorical_peak_table_{k}_{r}", "table_row_bg_grey")
            for n in range(0, 3):
                try:
                    dpg.add_text(f"{z_list[r*3+n]}+")
                except IndexError:
                    pass          
        
        with dpg.table_row(parent = f"theorical_peak_table_{k}"):
            for n in range(0, 3):
                try:
                    dpg.add_text(f"{mz_list[r*3+n]:.2f}")
                except IndexError:
                    pass
     

def update_peak_starting_points(sender = None, app_data= None, user_data:RenderCallback = None):
    spectrum = user_data.spectrum 
    low_limit = float(dpg.get_value("lower_bound")) /100
    high_limit = float(dpg.get_value("upper_bound")) /100
    print(low_limit, high_limit)

    if dpg.get_value("show_centers"):
        center_width = dpg.get_value("center_width") /100
        for peak in spectrum.peaks:
            if not spectrum.peaks[peak].fitted:
                continue
            apex = spectrum.peaks[peak].x0_refined
            sigma_L = spectrum.peaks[peak].sigma_L
            sigma_R = spectrum.peaks[peak].sigma_R
            spectrum.peaks[peak].start_range = (apex - sigma_L* center_width, apex + sigma_R * center_width)    
    else:

        for peak in spectrum.peaks:
            if not spectrum.peaks[peak].fitted:
                continue

            A= spectrum.peaks[peak].A_refined
            apex = spectrum.peaks[peak].x0_refined
            start = apex -  spectrum.peaks[peak].sigma_L
            
            while  bi_gaussian(start, spectrum.peaks[peak].A_refined, apex, spectrum.peaks[peak].sigma_L, spectrum.peaks[peak].sigma_R) > high_limit * A:       
                start -= 0.01
            start10pcs = start
            while  bi_gaussian(start, spectrum.peaks[peak].A_refined, apex, spectrum.peaks[peak].sigma_L, spectrum.peaks[peak].sigma_R) > low_limit * A:       
                start -= 0.01
            start1pcs = start
            spectrum.peaks[peak].start_range = (start1pcs, start10pcs)
        
    redraw_blocks(user_data)

def redraw_blocks(render_callback:RenderCallback):
    spectrum = render_callback.spectrum

    # Delete previous peaks and anotation
    for alias in dpg.get_aliases():
        if alias.startswith("fitted_peak_matching") or alias.startswith("peak_annotation_matching"):
            dpg.delete_item(alias)   
    
    for peak in spectrum.peaks:
        if not spectrum.peaks[peak].fitted:
            continue

        start1pcs, start10pcs = spectrum.peaks[peak].start_range
        mid = (start10pcs + start1pcs) / 2
        thick = start10pcs - start1pcs


        # non matched blocks
        dpg.draw_line((mid, 25), (mid, -25), parent="peak_matching_plot", color=(246, 32, 24,128), thickness=thick, tag=f"fitted_peak_matching_{peak}")
        dpg.add_plot_annotation(label=f"Peak {peak}", default_value=(mid, 0), offset=(15, 15), color=(100, 100, 100), clamped=False, parent="peak_matching_plot", tag=f"peak_annotation_matching_{peak}_gray")

        for k in range(0, len(render_callback.mz_lines)):
            mz_lines = render_callback.mz_lines[k]
            color = colors_list[k]

            # find matched blocks
            for z_mz in mz_lines:
                if z_mz[1] > start1pcs and z_mz[1] < start10pcs:
                    dpg.delete_item(f"fitted_peak_matching_{peak}")
                    dpg.draw_line((mid, 25), (mid, -25), parent="peak_matching_plot", color=color, thickness=thick, tag=f"fitted_peak_matching_{peak}")
                    
                    #dpg.delete_item(f"peak_annotation_matching_{k}_{z_mz[0]}")
                    if not dpg.does_alias_exist(f"peak_annotation_matching_{k}_{z_mz[0]}"):
                        x0_fit = spectrum.peaks[peak].x0_refined
                        center_slice = render_callback.spectrum.baseline_corrected[:,1][(render_callback.spectrum.baseline_corrected[:,0] > x0_fit - 0.5) & (render_callback.spectrum.baseline_corrected[:,0] < x0_fit + 0.5)]
                        median = np.median(center_slice)    if len(center_slice) > 0 else 0
                        dpg.add_plot_annotation(label=f"{z_mz[0]}+", default_value=(x0_fit, median), offset=(15, -15), color=color, clamped=False, parent="peak_matching_plot", tag=f"peak_annotation_matching_{k}_{z_mz[0]}")

    # Draw the line annotations of unmatched peaks      
    for k in range(0, len(render_callback.mz_lines)):
        mz_lines = render_callback.mz_lines[k]
        color = colors_list[k]
        for z_mz in mz_lines:
            if not dpg.does_alias_exist(f"peak_annotation_matching_{k}_{z_mz[0]}"):           
                dpg.add_plot_annotation(label=f"{z_mz[0]}+", default_value=(z_mz[1], 0), offset=(-15, -15-k*15), color=colors_list[k], clamped=False, parent="peak_matching_plot", tag=f"peak_annotation_matching_{k}_{z_mz[1]}")
    
    matching_quality(render_callback)
    mass_difference(render_callback)

def matching_quality(render_callback:RenderCallback):
    spectrum = render_callback.spectrum

    for k in range(0, len(render_callback.mz_lines)):
        mz_lines = render_callback.mz_lines[k]
        squares = []
        
        for z_mz in mz_lines:
            for peak in spectrum.peaks:
                if not spectrum.peaks[peak].fitted:
                    continue

                start1pcs, start10pcs = spectrum.peaks[peak].start_range
                if z_mz[1] > start1pcs and z_mz[1] < start10pcs:
                    center = (start1pcs + start10pcs) / 2
                    square_distance = (z_mz[1] - center)**2
                    squares.append(square_distance)
        
        if len(squares) > 1:
            rmsd = np.sqrt(np.mean(squares))
            score = int((1 / rmsd) * 100) + (len(squares) * 20)
            dpg.set_value(f"rmsd_{k}", f"{len(squares)} - RMSD: {rmsd:.2f} Score: {score}")
        
        elif len(squares) == 1:
            dpg.set_value(f"rmsd_{k}", f"Only 1 peak match")
        else:
            dpg.set_value(f"rmsd_{k}", f"No matching peaks")

def mass_difference(render_callback:RenderCallback):

    mw_set1 =dpg.get_value("molecular_weight_0")
    for i in range(1, 5):
        mw = dpg.get_value(f"molecular_weight_{i}")
        diff = mw - mw_set1
        dpg.set_value(f"MW_diff_{i}", f"d{diff}")


    
            
        
                    
                

        
