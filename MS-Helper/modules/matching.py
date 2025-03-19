from typing import List, Tuple
import dearpygui.dearpygui as dpg
from modules.data_structures import MSData, peak_params
from modules.helpers import bi_gaussian
import numpy as np
from modules.render_callback import RenderCallback
from modules.var import colors_list

def matching_window(render_callback:RenderCallback):
    spectrum = render_callback.spectrum
    with dpg.window(label="Peak matching", width=1430, height=850, pos=(0,110), no_close=True, no_move=True, no_resize=True, tag="Peak matching", collapsed=True):
        # Create a plot for the raw data
        with dpg.plot(label="Peak matching", width=1430, height=700, tag="peak_matching_plot") as plot2:
            # Add x and y axes
            dpg.add_plot_axis(dpg.mvXAxis, label="m/z", tag="x_axis_plot3")
            dpg.add_plot_axis(dpg.mvYAxis, label="Y Axis", tag="y_axis_plot3")
            dpg.add_line_series(spectrum.baseline_corrected[:,0], spectrum.baseline_corrected[:,1], label="Corrected Data Series", parent="y_axis_plot3", tag="corrected_series_plot3")
        
        with dpg.group(horizontal=True, horizontal_spacing= 25):
            with dpg.group(horizontal=False):
                dpg.add_button(label="Show fitted peaks", callback=update_peak_starting_points, user_data=render_callback)
                dpg.add_text("Peaks Start:")
                dpg.add_input_int(label="Lower  %", default_value=1,min_value=1 , max_value=100, tag="lower_bound", width=100)
                dpg.add_input_int(label="Upper  %", default_value=20, min_value=1 , max_value=100, tag="upper_bound", width=100)
                dpg.add_checkbox(label="Show Centers instead", default_value=False, tag="show_centers", callback=update_peak_starting_points, user_data=render_callback)
                dpg.add_input_int(label="Width", default_value=1, min_value=1 , max_value=100, tag="center_width", width=100)
                
            for i in range(5):
                with dpg.child_window(height=200, width=220, tag = f"theorical_peaks_window_{i}"):
                    with dpg.theme(tag=f"theme_peak_window_{i}"):
                        with dpg.theme_component():
                            dpg.add_theme_color(dpg.mvThemeCol_Border, colors_list[i], category=dpg.mvThemeCat_Core)
                    
                    dpg.add_text(f"Peak Set {i}", tag=f"rmsd_{i}")
                    
                    with dpg.group(horizontal=True):
                        dpg.add_input_int(label="MW", default_value=549000, tag=f"molecular_weight_{i}", step = 100, width = 125, callback=draw_mz_lines, user_data=(render_callback, i))
                        dpg.add_text("", tag = f"MW_diff_{i}")
                    
                    dpg.add_input_int(label="Charges", default_value=52, tag=f"charges_{i}", width = 125, callback=draw_mz_lines,  user_data=(render_callback, i))
                    dpg.add_input_int(label="# Peaks", default_value=5, tag=f"nb_peak_show_{i}",step = 1, width = 125, callback=draw_mz_lines, user_data=(render_callback, i))
                    dpg.add_table(header_row=False, row_background=True, tag=f"theorical_peak_table_{i}")
                    dpg.bind_item_theme(f"theorical_peaks_window_{i}", f"theme_peak_window_{i}")


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