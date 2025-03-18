import dearpygui.dearpygui as dpg
from modules.data_structures import MSData
from modules.finding import peaks_finder_callback, add_peak
from modules.fitting import  draw_fitted_peaks, run_fitting
from modules.dpg_draw import *
from modules.var import colors_list
from modules.render_callback import RenderCallback


spectrum = MSData()
spectrum.import_csv(rf"D:\MassSpec\Um_2-1_1x.csv")
#spectrum.import_csv(rf"D:\MassSpec\Um_data.csv")
#spectrum.import_csv(rf"D:\MassSpec\CS_RBC_alone.csv")
#spectrum.import_csv(rf"D:\MassSpec\CR_aloneCID.csv")

# Create a Dear PyGui context
dpg.create_context()
render_callback = RenderCallback(spectrum)

with dpg.window(label="Control", width=1430, height=-1, no_close=True, no_collapse=True, no_move=True, no_resize=True, no_title_bar=True, tag="Control"):
    dpg.set_primary_window( "Control", True)
    # Add a slider to adjust the window length
    dpg.add_input_text(width=-1, height=50, multiline=True, readonly=True, tracked=True, track_offset=1.0, tag="message_box")
# Create a window
with dpg.window(label="Data Filtering and peak finding",width=1430, pos=(0,60), height=1000, no_close=True, no_move=True, no_resize=True, tag="Data Filtering", collapsed=False):
    with dpg.group(horizontal=True, horizontal_spacing= 50):
        dpg.add_text("Data Clipping:")
        min_value = min(spectrum.original_data[:,0])
        max_value = max(spectrum.original_data[:,0])
        dpg.add_slider_int(label="Data clipping left", width=400, default_value=min_value, min_value=min_value, max_value=max_value, tag="L_data_clipping", callback=data_clipper, user_data=spectrum)
        dpg.add_slider_int(label="Data clipping right", width=400, default_value=max_value, min_value=min_value, max_value=max_value, tag="R_data_clipping", callback=data_clipper, user_data=spectrum)
   
    # Create a plot for the data
    with dpg.plot(label="Data Filtering", width=1430, height=600, tag="data_plot") as plot1:
        # Add x and y axes
        x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="m/z", tag= "x_axis_plot1")
        y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Y Axis", tag = "y_axis_plot1")
        
        w_x = spectrum.working_data[:,0].tolist()
        dpg.add_line_series(w_x, spectrum.working_data[:,1].tolist(), label="Original Data Series", parent=y_axis, tag="original_series")
        dpg.add_line_series(w_x, spectrum.get_filterd_data(50), label="Filtered Data Series", parent=y_axis, tag="filtered_series")
        dpg.add_line_series(spectrum.baseline[:,0].tolist(), spectrum.baseline[:,1].tolist(), label="Snip Baseline", parent=y_axis, tag="baseline")  

    with dpg.group(horizontal=True, horizontal_spacing= 50):
        with dpg.child_window(height=230, width=300):     
            dpg.add_text("Data Filtering:")
            dpg.add_text("Smoothing window:")
            dpg.add_slider_int(label="", default_value=925, min_value=3, max_value=1000, width=250, callback=filter_data, user_data=spectrum, tag="smoothing_window")
            dpg.add_text("")
            dpg.add_text("Baseline estimation:")
            with dpg.group(horizontal=True, horizontal_spacing= 50):
                dpg.add_button(label="Toggle Baseline", callback=toggle_baseline, user_data=spectrum)
                dpg.add_button(label="Update Baseline",  callback=spectrum.request_baseline_update, user_data=spectrum)
            dpg.add_text("Baseline window:") 
            dpg.add_slider_int(label="", default_value=500, min_value=10, max_value=2000, width=250, callback=spectrum.request_baseline_update, user_data=spectrum, tag="baseline_window")

        with dpg.child_window(height=230, width=300):           
            dpg.add_text("Peak detection:")
            dpg.add_text("Peak detection threshold:")
            dpg.add_slider_int(label="", width=200, default_value=100, min_value=1, max_value=300, tag="peak_detection_threshold")
            dpg.add_text("Peak detection width:")
            dpg.add_slider_int(label="", width=200, default_value=20, min_value=2, max_value=100, tag="peak_detection_width")
            dpg.add_text("Peak detection distance:")
            dpg.add_slider_int(label="", width=200, default_value=100, min_value=1, max_value=1000, tag="peak_detection_distance")
            dpg.add_button(label="Find Peaks", callback=peaks_finder_callback, user_data=render_callback)
        
        with dpg.child_window(height=230, width=300):
            dpg.add_text("Peaks:")
            dpg.add_table(header_row=True, tag="found_peak_table")
            dpg.add_table_column(label="Peak Label", parent="found_peak_table")
            dpg.add_table_column(label="Use", parent="found_peak_table")

        with dpg.child_window(height=230, width=300):
            dpg.add_text("User Peaks:")
            dpg.add_button(label="Add Peak", callback=add_peak, user_data=render_callback)
            dpg.add_table(header_row=True, tag="user_peak_table")
            dpg.add_table_column(label="Peak Label", parent="user_peak_table")
            dpg.add_table_column(label="Use", parent="user_peak_table")

with dpg.window(label="Peak fitting", width=1430, height=800, pos=(0,85), no_close=True, no_move=True, no_resize=True, tag="Peak fitting", collapsed=True):
    # Create a plot for the raw data
    with dpg.plot(label="Gaussian Fit", width=1430, height=600, tag="gaussian_fit_plot") as plot2:
        # Add x and y axes
        dpg.add_plot_axis(dpg.mvXAxis, label="m/z", tag="x_axis_plot2")
        dpg.add_plot_axis(dpg.mvYAxis, label="Y Axis", tag="y_axis_plot2")        
        # Add the raw data series to the plot
        dpg.add_line_series(spectrum.baseline_corrected[:,0], spectrum.baseline_corrected[:,1], label="Corrected Data Series", parent="y_axis_plot2", tag="corrected_series_plot2")
        #dpg.add_line_series(spectrum.baseline_corrected[:,0], [0]*len(spectrum.baseline_corrected[:,0]), label="Residual", parent="y_axis_plot2", tag="residual")

    with dpg.group(horizontal=True, horizontal_spacing= 50):
        #dpg.add_button(label="Multi Fit Gaussians", callback=multi_bigaussian_fit, user_data=spectrum)
        dpg.add_button(label="Windowed Multi Bi Gaussian Deconvolution", callback=run_fitting, user_data=spectrum)
        dpg.add_loading_indicator(style=5,radius=3, show=False, tag="Fitting_indicator")
        dpg.add_text("", tag="Fitting_indicator_text")
    dpg.add_button(label="Redraw Peaks", callback=draw_fitted_peaks, user_data=spectrum)

    with dpg.child_window(height=100, width=900, tag="peak_table_window"):
        with dpg.table(header_row=True, tag="peak_table"):
            dpg.add_table_column(label="Peak Label")
            dpg.add_table_column(label="Peak Start")
            dpg.add_table_column(label="Peak Apex")
            dpg.add_table_column(label="Peak Integral")

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
                #dpg.add_table(header_row=True, tag=f"theorical_peak_table_{i}_2")
                dpg.bind_item_theme(f"theorical_peaks_window_{i}", f"theme_peak_window_{i}")


        

# Import the custom theme
from modules.dpg_style import general_theme, data_theme
dpg.bind_theme(general_theme)
dpg.bind_item_theme("original_series", data_theme)
dpg.bind_item_theme("filtered_series", "filtered_data_theme")
dpg.bind_item_theme("baseline", "baseline_theme")
dpg.bind_item_theme("corrected_series_plot2", data_theme)
dpg.bind_item_theme("corrected_series_plot3", data_theme)

### Auto start
dpg.set_value("L_data_clipping", 9500)
dpg.set_value("R_data_clipping", 11700)
dpg.set_value("peak_detection_threshold", 50)
dpg.set_value("peak_detection_width", 8)
dpg.set_value("peak_detection_distance", 30)
dpg.set_value("baseline_window", 1000)
toggle_baseline(None,None, spectrum)
import time
time.sleep(1)
data_clipper(None, None, spectrum)
spectrum.baseline_need_update = True
#peaks_finder_callback(None, None, spectrum)
#run_fitting(None, None, spectrum)
#update_peak_starting_points(None, None, render_callback)
#multi_bigaussian_fit(None, None,spectrum)
# Create a viewport and show the plot
dpg.create_viewport(title='Multi Bi Gaussian Fit', width=1450, height=1000, x_pos=0, y_pos=0)
dpg.focus_item("Data Filtering")
dpg.focus_item("Peak fitting")
data_clipper(None, None, spectrum)
#dpg.focus_item("Peak matching")
dpg.show_style_editor()
dpg.show_metrics()
dpg.setup_dearpygui()
dpg.show_viewport()

while(dpg.is_dearpygui_running()):

    render_callback.execute()
    dpg.render_dearpygui_frame()  

dpg.destroy_context()