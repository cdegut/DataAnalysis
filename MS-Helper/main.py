import dearpygui.dearpygui as dpg
from modules.msdata_class import MSData, peak_params
from modules.funct import data_clipper, filter_data, toggle_baseline, correct_baseline, peaks_finder, multi_bigaussian_fit, draw_fitted_peaks, draw_mz_lines


spectrum = MSData()
spectrum.import_csv(rf"D:\MassSpec\Um_2-1_1x.csv")


# Create a Dear PyGui context
dpg.create_context()

with dpg.window(label="Control", width=1430, height= 10, no_close=True, no_collapse=True, no_move=True, no_resize=True, no_title_bar=True):
    # Add a slider to adjust the window length
    with dpg.group(horizontal=True, horizontal_spacing= 50):
        dpg.add_text("Data Clipping:")
        min_value = min(spectrum.original_data[:,0])
        max_value = max(spectrum.original_data[:,0])
        dpg.add_slider_int(label="Data clipping left", width=400, default_value=min_value, min_value=min_value, max_value=max_value, tag="L_data_clipping", callback=data_clipper, user_data=spectrum)
        dpg.add_slider_int(label="Data clipping right", width=400, default_value=max_value, min_value=min_value, max_value=max_value, tag="R_data_clipping", callback=data_clipper, user_data=spectrum)
    dpg.add_input_text(width=-1, height=50, multiline=True, readonly=True, tracked=True, track_offset=1.0, tag="message_box")
# Create a window
with dpg.window(label="Data Filtering and peak finding",width=1430, pos=(0,100), no_close=True, no_move=True, no_resize=True, tag="Data Filtering", collapsed=True):
    
    # Create a plot for the data
    with dpg.plot(label="Data Filtering", width=1430, height=600, tag="data_plot") as plot1:
        # Add x and y axes
        x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="m/z", tag= "x_axis_plot1")
        y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Y Axis", tag = "y_axis_plot1")
        
        dpg.add_line_series(spectrum.working_data[:,0].tolist(), spectrum.working_data[:,1].tolist(), label="Original Data Series", parent=y_axis, tag="original_series")
        dpg.add_line_series(spectrum.filtered[:,0].tolist(), spectrum.filtered[:,1].tolist(), label="Filtered Data Series", parent=y_axis, tag="filtered_series")
        dpg.add_line_series(spectrum.baseline[:,0].tolist(), spectrum.baseline[:,1].tolist(), label="Snip Baseline", parent=y_axis, tag="baseline")  
        
    dpg.add_text("Data Filtering:")
    dpg.add_slider_int(label="Smoothing Window", default_value=100, min_value=3, max_value=1000, callback=filter_data, user_data=spectrum, tag="smoothing_window")
    dpg.add_text("Baseline estimation:")
    with dpg.group(horizontal=True, horizontal_spacing= 50):
        dpg.add_button(label="Toggle Baseline", callback=toggle_baseline, user_data=spectrum)
        dpg.add_slider_int(label="Baseline window", default_value=500, min_value=100, max_value=1000, tag="baseline_window")
        dpg.add_button(label="Update Baseline", callback=correct_baseline, user_data=spectrum)  
      
    dpg.add_text("Peak detection:")
    dpg.add_slider_int(label="Peak detection distance from baseline", default_value=100, min_value=1, max_value=300, tag="peak_detection_threshold")
    dpg.add_slider_int(label="Peak width", default_value=200, min_value=1, max_value=600, tag="peak_detection_width")
    dpg.add_slider_int(label="Peak min distance", default_value=1000, min_value=1, max_value=1000, tag="peak_detection_distance")
    dpg.add_button(label="Find Peaks", callback=peaks_finder, user_data=spectrum)


with dpg.window(label="Peak fitting", width=1430, height=800, pos=(0,125), no_close=True, no_move=True, no_resize=True, collapsed=True):
    # Create a plot for the raw data
    with dpg.plot(label="Gaussian Fit", width=1430, height=600, tag="gaussian_fit_plot") as plot2:
        # Add x and y axes
        dpg.add_plot_axis(dpg.mvXAxis, label="m/z", tag="x_axis_plot2")
        dpg.add_plot_axis(dpg.mvYAxis, label="Y Axis", tag="y_axis_plot2")
        
        # Add the raw data series to the plot
        dpg.add_line_series(spectrum.baseline_corrected_clipped[:,0], spectrum.baseline_corrected_clipped[:,1], label="Corrected Data Series", parent="y_axis_plot2", tag="corrected_series_plot2")
    
    with dpg.group(horizontal=True, horizontal_spacing= 50):
        dpg.add_button(label="Multi Fit Gaussians", callback=multi_bigaussian_fit, user_data=spectrum)
        dpg.add_button(label="Redraw Peaks", callback=draw_fitted_peaks, user_data=spectrum)

    with dpg.child_window(height=100, width=900, tag="peak_table_window"):
        with dpg.table(header_row=True, tag="peak_table"):
            dpg.add_table_column(label="Peak Label")
            dpg.add_table_column(label="Peak Start")
            dpg.add_table_column(label="Peak Apex")
            dpg.add_table_column(label="Peak Integral")

with dpg.window(label="Peak matching", width=1430, height=800, pos=(0,150), no_close=True, no_move=True, no_resize=True):
    # Create a plot for the raw data
    with dpg.plot(label="Peak matching", width=1430, height=600, tag="peak_matching_plot") as plot2:
        # Add x and y axes
        dpg.add_plot_axis(dpg.mvXAxis, label="m/z", tag="x_axis_plot3")
        dpg.add_plot_axis(dpg.mvYAxis, label="Y Axis", tag="y_axis_plot3")
        dpg.add_line_series(spectrum.baseline_corrected_clipped[:,0], spectrum.baseline_corrected_clipped[:,1], label="Corrected Data Series", parent="y_axis_plot3", tag="corrected_series_plot3")
    
    with dpg.group(horizontal=True, horizontal_spacing= 50):
        with dpg.child_window(height=110, width=400):
            dpg.add_text("Theorical Peaks set 1:")
            dpg.add_input_int(label="Molecular Weight", default_value=549000, tag="molecular_weight_0", step = 100, callback=draw_mz_lines, user_data=0)
            dpg.add_input_int(label="Charges", default_value=52, tag="charges_0", callback=draw_mz_lines, user_data=0)
            dpg.add_input_int(label="Peaks to show", default_value=5, tag="nb_peak_show_0", callback=draw_mz_lines, user_data=0)
        with dpg.child_window(height=110, width=400):
            dpg.add_text("Theorical Peaks set 2:")
            dpg.add_input_int(label="Molecular Weight", default_value=549000, tag="molecular_weight_1", step = 100, callback=draw_mz_lines, user_data=1)
            dpg.add_input_int(label="Charges", default_value=52, tag="charges_1", callback=draw_mz_lines, user_data=1)
            dpg.add_input_int(label="Peaks to show", default_value=5, tag="nb_peak_show_1", callback=draw_mz_lines, user_data=1)
        with dpg.child_window(height=110, width=400):
            dpg.add_text("Theorical Peaks set 3:")
            dpg.add_input_int(label="Molecular Weight", default_value=549000, tag="molecular_weight_2", step = 100, callback=draw_mz_lines, user_data=2)
            dpg.add_input_int(label="Charges", default_value=52, tag="charges_2", callback=draw_mz_lines, user_data=2)
            dpg.add_input_int(label="Peaks to show", default_value=5, tag="nb_peak_show_2", callback=draw_mz_lines, user_data=3)

# Import the custom theme
from modules.dpg_style import general_theme, data_theme
dpg.bind_theme(general_theme)
dpg.bind_item_theme("original_series", data_theme)
dpg.bind_item_theme("corrected_series_plot2", data_theme)
dpg.bind_item_theme("corrected_series_plot3", data_theme)


# Create a viewport and show the plot
dpg.create_viewport(title='Multi Bi Gaussian Fit', width=1450, height=1000)
dpg.focus_item("Data Filtering")
dpg.show_style_editor()
dpg.show_metrics()
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()