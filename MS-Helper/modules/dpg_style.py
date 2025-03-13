import dearpygui.dearpygui as dpg

with dpg.theme() as general_theme:
    with dpg.theme_component(dpg.mvAll):
        #dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (131, 184, 198), category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots)

with dpg.theme() as data_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1, category=dpg.mvThemeCat_Plots)
        dpg.add_theme_color(dpg.mvPlotCol_Line, (100, 100, 100), category=dpg.mvThemeCat_Plots)