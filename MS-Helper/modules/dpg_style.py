import dearpygui.dearpygui as dpg
from modules.var import colors_list

with dpg.theme() as general_theme:
    with dpg.theme_component(dpg.mvAll):
        #dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (131, 184, 198), category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots)

with dpg.theme() as data_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1, category=dpg.mvThemeCat_Plots)
        dpg.add_theme_color(dpg.mvPlotCol_Line, (100, 100, 100), category=dpg.mvThemeCat_Plots)
i = 0
for color in colors_list:
    with dpg.theme(tag = f"mz_line_theme_{i}"):
        with dpg.theme_component(dpg.mvInfLineSeries):
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
    i += 1

with dpg.theme(tag = f"matching_lines"):
    with dpg.theme_component(dpg.mvInfLineSeries):
        dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 6, category=dpg.mvThemeCat_Plots)
        dpg.add_theme_color(dpg.mvPlotCol_Line, (246, 56, 45, 64), category=dpg.mvThemeCat_Plots)