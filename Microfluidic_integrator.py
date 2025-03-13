
import cv2 
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PIL.Image, PIL.ImageGrab
import customtkinter as ctk
import glob
from matplotlib.colors import to_rgb
import gc
from memory_profiler import profile

#####################################
###### Image handeling functions #####



def pltcolor_to_cv2(color):
    rgb = to_rgb(color)
    rgb = [int(c*255) for c in rgb ]
    bgr = (rgb[2], rgb[1], rgb[0])
    return bgr

def get_background_values(image, ROI ):
    background = image[int(ROI[1]):int(ROI[1]+ROI[3]),int(ROI[0]):int(ROI[0]+ROI[2])]
    (back_b_channel, back_g_channel, back_r_channel) = cv2.split(background)
    b = (np.mean(back_b_channel),np.std(back_b_channel)) 
    g = (np.mean(back_g_channel),np.std(back_g_channel)) 
    r = (np.mean(back_r_channel),np.std(back_r_channel))
    return (b,g,r)

def background_supression_display(image, thresolds):
    
    (b_channel, g_channel, r_channel) = cv2.split(image)

    b_thresold_cut, g_thresold_cut,  r_thresold_cut = bgr_channel_thresolding(b_channel, g_channel, r_channel, thresolds)
    
    rebuild = cv2.merge([b_thresold_cut.astype("uint8"), g_thresold_cut.astype("uint8") , r_thresold_cut.astype("uint8")])

    return rebuild

def bgr_channel_thresolding(b_channel, g_channel, r_channel, thresolds):
        
        b_thresold_cut = b_channel.astype(int) - int(thresolds[0])
        b_thresold_cut[b_thresold_cut < 0 ] = 0
        g_thresold_cut = g_channel.astype(int) - int(thresolds[1])
        g_thresold_cut[g_thresold_cut < 0 ] = 0
        r_thresold_cut = r_channel.astype(int) - int(thresolds[2])
        r_thresold_cut[r_thresold_cut < 0 ] = 0
        return b_thresold_cut, g_thresold_cut,  r_thresold_cut

def highlight_supression_display(image, thresold=200):
    (b_channel, g_channel, r_channel) = cv2.split(image)
    mask = np.add(b_channel.astype(int), g_channel.astype(int), r_channel.astype(int))

    b_channel[mask >= thresold] = 0
    g_channel[mask >= thresold] = 0   
    r_channel[mask >= thresold] = 0

    clipped = cv2.merge((b_channel.astype("uint8"), g_channel.astype("uint8"), r_channel.astype("uint8")))

    return clipped


def cv2_draw_ROIs(image, bg_ROI=None, ROIs=None, colors=None, bg_color=(205,0,0) ):
    font = cv2.FONT_HERSHEY_DUPLEX
    if ROIs is not None:
        
        i = 1
        for ROI in ROIs:
            if colors is not None:
                color = colors[i-1]
            else: 
                color = (0,0,255)
            image = cv2.rectangle( image, (ROI[0], ROI[1]), (ROI[0]+ROI[2], ROI[1]+ROI[3]), color, 4)
            image = cv2.putText(image, str(i), (ROI[0]+ROI[2], ROI[1]), font, fontScale=4, color=color, thickness=6)
            i = i+1
    if bg_ROI:
        image = cv2.rectangle( image, (bg_ROI[0], bg_ROI[1]), (bg_ROI[0]+bg_ROI[2], bg_ROI[1]+bg_ROI[3]), bg_color, 4)
        image = cv2.putText(image, "Bg", (bg_ROI[0]+bg_ROI[2], bg_ROI[1]), font, fontScale=4, color=bg_color, thickness=6 )
    
    return image

def process_ROIs(image, ROIs, background_thresolds = None, highlight_thresold = None):

    (b_channel, g_channel, r_channel) = cv2.split(image)

    if highlight_thresold is not None:
        mask = np.add(b_channel.astype(int), g_channel.astype(int), r_channel.astype(int))
    
    if background_thresolds is not None:
         b_channel, g_channel, r_channel = bgr_channel_thresolding(b_channel, g_channel, r_channel, background_thresolds)

    else:
        b_channel = b_channel.astype(int)
        g_channel = g_channel.astype(int)
        r_channel = r_channel.astype(int)

    if highlight_thresold is not None:
        b_channel[mask >= highlight_thresold] = -1
        g_channel[mask >= highlight_thresold] = -1   
        r_channel[mask >= highlight_thresold] = -1
    
    ROIs_data = []
    for ROI in ROIs:
        b_crop = b_channel[int(ROI[1]):int(ROI[1]+ROI[3]),int(ROI[0]):int(ROI[0]+ROI[2])]
        g_crop = g_channel[int(ROI[1]):int(ROI[1]+ROI[3]),int(ROI[0]):int(ROI[0]+ROI[2])]
        r_crop = r_channel[int(ROI[1]):int(ROI[1]+ROI[3]),int(ROI[0]):int(ROI[0]+ROI[2])]


        index_b = np.argwhere(b_crop==-1)
        b_crop = np.delete(b_crop, index_b)
        b = np.mean(b_crop)
        b_std = np.std(b_crop)

        index_g = np.argwhere(g_crop==-1)
        g_crop = np.delete(g_crop, index_g)
        g_crop[g_crop >=0 ]
        g = np.mean(g_crop)
        g_std = np.std(g_crop)

        index_r = np.argwhere(r_crop==-1)
        r_crop = np.delete(r_crop, index_r)
        r_crop[r_crop >=0 ]
        r = np.mean(r_crop)
        r_std = np.std(r_crop)

        
        l = [(b, b_std),(g, g_std), (r, r_std)]
        ROIs_data.append(l)
    
    return ROIs_data
    
def folder_process(thresolds, clip, ROIs, path="D:\\microscope_data\\time-lapse\\231030_154809\\"):
    #cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Image", 800, 600)

    data = []

    for image in glob.glob(f"{path}\\*.png"):
        current_image = cv2.imread(image)
        
        ROIs_data = process_ROIs(current_image, ROIs)
       
        data.append(ROIs_data)
        

        current_image = cv2_draw_ROIs(current_image, bg_ROI = None, ROIs=ROIs)
        #cv2.imshow("Image", current_image)
        #cv2.waitKey(10)
    data_array = np.array(data)
    #cv2.destroyWindow("Image")
    return data_array
        
        
################################################################################
################################################################################
#################### TK app ####################################################


class Interface(ctk.CTkFrame):
    def __init__(self, Tk_root, path):
        ctk.CTkFrame.__init__(self, Tk_root)
        self.Tk_root = Tk_root

        self.redraw_job = None

        self.default_bgr_sliders = (2,0,2)
        self.default_clip_slider = 450

        self.images_list = glob.glob(f"{path}\\*.png")
        self.total_images = len(self.images_list)
        self.second_per_image = 5
        

        self.raw_image= cv2.imread(self.images_list[0])
        self.current_image = 1
        self.display_image = None
        self.display = None
        self.image_as_label = None

        self.ROIs = None

        self.bg_ROI = None
        self.bg_thresolds = None

        self.color_list = ["crimson", "gold", "darkorchid", "darkgoldenrod"]
        self.bg_color = pltcolor_to_cv2("mediumblue")
        self.bgr_color_list = [pltcolor_to_cv2(color) for color in self.color_list]

        self.data_array = np.zeros((self.total_images,1,3,2))

        self.init_window()
    
    ########################
    #### Interface bits ####
    ########################

    def init_window(self):

        self.matplotlib_make_plot()
        self.filters_sliders_draw((820, 40))
        self.contrast_brightness_sliders_draw((820, 360))
        self.images_slider_draw((405,600), 750)
        button_position = (840, 400)

        bg_select_button = ctk.CTkButton(master=self.Tk_root, text="Select Background Area", width=180, command=self.cv2_bg_selector)
        bg_select_button.place(x=button_position[0], y=button_position[1]+40)
        bg_clear_button = ctk.CTkButton(master=self.Tk_root, text="Clear Background Area", width=180, command=self.clear_bg)
        bg_clear_button.place(x=button_position[0], y=button_position[1]+80)
                
        ROI_select_button = ctk.CTkButton(master=self.Tk_root, text="Select Integration Regions",width=180, command=self.cv2_ROIs_selector)
        ROI_select_button.place(x=button_position[0], y=button_position[1]+120)


        process_button = ctk.CTkButton(master=self.Tk_root, text="Process Folder Data",width=180, command=self.process_folder, fg_color="green")
        process_button.place(x=button_position[0], y=button_position[1]+180)

        self.label_update()

        self.ROIs_label = ctk.CTkLabel(master = self.Tk_root, justify="left", text = "ROIs integration data:")
        self.ROIs_label.place(x=820, y = 700)

        self.draw_image()
        self.matplotlib_plot()

    def images_slider_draw(self, position, lengh):
        totalimages = len(self.images_list)
        self.images_slider = ctk.CTkSlider(master = self.Tk_root, from_=0, to=totalimages-1, number_of_steps=totalimages-1, width=lengh -80, command=self.load_image, orientation=ctk.HORIZONTAL)
        self.images_slider.place(x=position[0], y = position[1]+30, anchor=ctk.N)
        self.images_slider.set(0)
        self.image_label = ctk.CTkLabel(master = self.Tk_root, text = "Image #: 1")
        self.image_label.place(x=position[0]-310, y = position[1]) 
        self.previous_image = ctk.CTkButton(master=self.Tk_root, text="Prev",width=60, command=lambda: self.load_image(self.current_image - 1))
        self.previous_image.place(x=position[0]-370, y = position[1]+25, anchor=ctk.N)
        self.next_image = ctk.CTkButton(master=self.Tk_root, text="Next",width=60, command=lambda: self.load_image(self.current_image + 1))
        self.next_image.place(x=position[0]+370, y = position[1]+25, anchor=ctk.N)

        self.forward = ctk.CTkButton(master=self.Tk_root, text="Play",width=60, command=self.play)
        self.forward.place(x=position[0]+450, y = position[1]+25, anchor=ctk.N)        

    def filters_sliders_draw(self, position):
        
        self.b_slider = ctk.CTkSlider(master = self.Tk_root, from_=-10, to=10, number_of_steps=20,  command=self.redraw_clear_data, orientation=ctk.HORIZONTAL)
        self.g_slider = ctk.CTkSlider(master = self.Tk_root, from_=-10, to=10, number_of_steps=20, command=self.redraw_clear_data, orientation=ctk.HORIZONTAL)
        self.r_slider = ctk.CTkSlider(master = self.Tk_root, from_=-10, to=10,  number_of_steps=20, command=self.redraw_clear_data, orientation=ctk.HORIZONTAL)
        
        title_label = ctk.CTkLabel(master = self.Tk_root, justify="left", text = "Adjust filter: \nwill change data integration values")
        title_label.place(x=position[0]-10, y = position[1]-30)
        self.b_label = ctk.CTkLabel(master = self.Tk_root, text = "Blue:")
        self.b_label.place(x=position[0], y = position[1])
        self.b_slider.place(x=position[0]+10, y = position[1]+30)
        
        self.g_label = ctk.CTkLabel(master = self.Tk_root, text = "Green:")
        self.g_label.place(x=position[0], y = position[1]+60)
        self.g_slider.place(x=position[0]+10, y = position[1]+90)
        
        self.r_label = ctk.CTkLabel(master = self.Tk_root, text = "Red:")
        self.r_label.place(x=position[0], y = position[1]+120)
        self.r_slider.place(x=position[0]+10, y = position[1]+150)
        
        self.b_slider.set(self.default_bgr_sliders[0])
        self.g_slider.set(self.default_bgr_sliders[1])
        self.r_slider.set(self.default_bgr_sliders[2])

        self.clip_slider = ctk.CTkSlider(master = self.Tk_root, from_=0, to=800, number_of_steps=40, command=self.redraw_clear_data,  orientation=ctk.HORIZONTAL)
        self.clip_label = ctk.CTkLabel(master = self.Tk_root, text = "Highlight clipping level:")
        self.clip_label.place(x=position[0], y = position[1]+180)
        self.clip_slider.place(x=position[0]+10, y = position[1]+210)
        self.clip_slider.set(self.default_clip_slider)
        
        default_button = ctk.CTkButton(master=self.Tk_root, text="Default",width=180, command=self.default)
        default_button.place(x=position[0]+10, y=position[1]+240)


    def contrast_brightness_sliders_draw(self, position):

        title_label = ctk.CTkLabel(master = self.Tk_root, justify="left", text = "Adjust display: \nwill *NOT* change data integration values")
        title_label.place(x=position[0]-10, y = position[1]-30)

        #self.contrast_label = ctk.CTkLabel(master = self.Tk_root, text = "Contrast:")
        #self.contrast_slider = ctk.CTkSlider(master = self.Tk_root, from_=0, to=256, number_of_steps=21,  command=self.redraw, orientation=ctk.HORIZONTAL)
        self.brightness_label = ctk.CTkLabel(master = self.Tk_root, text = "Brithness:")
        self.brigthness_slider = ctk.CTkSlider(master = self.Tk_root, from_=0, to=3, number_of_steps=300, command=self.redraw, orientation=ctk.HORIZONTAL)
        self.brigthness_slider.set(1)  
        #self.contrast_label.place(x=position[0], y = position[1]+60)
        #self.contrast_slider.place(x=position[0]+10, y = position[1]+90) 
        self.brightness_label.place(x=position[0], y = position[1])
        self.brigthness_slider.place(x=position[0]+10, y = position[1]+30)
    
    def default(self):
        self.b_slider.set(self.default_bgr_sliders[0])
        self.g_slider.set(self.default_bgr_sliders[1])
        self.r_slider.set(self.default_bgr_sliders[2])
        self.clip_slider.set(self.default_clip_slider)
    
        self.draw_image()
    
    def clear_bg(self):
        self.bg_ROI = None
        self.redraw()

    def label_update(self):
        self.b_label.configure(text= f"Blue: {self.b_slider.get()}")
        self.g_label.configure(text= f"Green: {self.g_slider.get()}")
        self.r_label.configure(text= f"Red: {self.r_slider.get()}")
        self.clip_label.configure(text= f"Highlight clipping level: {self.clip_slider.get()}")
        #self.contrast_label.configure(text= f"Contrast: {self.contrast_slider.get()}")
        self.brightness_label.configure(text= f"Brightness: {self.brigthness_slider.get():.2f}")
        self.after(100, self.label_update)
    
    def play(self):
        self.draw_PIL_image(self.current_image) 
        if self.current_image == self.total_images-1:
            print("end")
            return    
        self.load_image(self.current_image + 1)        
        self.play_job = self.after(100, self.play)

    def update_ROIs_label(self, ROIs_data):

        text = "ROIs integration data:\n"
        i = 1
        for ROI_data in ROIs_data:
            label = f"\n #{i}  B:{round(ROI_data[0][0],2)}  G:{round(ROI_data[1][0],2)}  R:{round(ROI_data[2][0],2)}"
            text = text + label
            i = i +1

        self.ROIs_label.configure(text = text)         

    ###########################
    #### Image handeling ######
    ###########################
    def load_image(self, x):
        i = int(x)
        self.images_slider.set(x)
        self.current_image = i
        self.raw_image= cv2.imread(self.images_list[i])
        self.image_label.configure(text=f"Image #: {i+1}")
        self.draw_image() 

    def clear_plot_and_data_array(self):
        try:
            self.data_array = np.zeros((self.total_images, len(self.ROIs), 3, 2))
        except:
            self.data_array = np.zeros((self.total_images, 1, 3, 2))

        self.data_array[ self.data_array==0 ] = np.nan
        self.matplotlib_make_plot()

    def redraw(self, x=None):        
        if self.redraw_job:           
            self.Tk_root.after_cancel(self.redraw_job)        
        self.redraw_job = self.after(50, self.draw_image)
        
    def redraw_clear_data(self, x=None):
        self.clear_plot_and_data_array()
        self.redraw()

    def draw_image(self):

        self.display_image = highlight_supression_display(self.raw_image, self.clip_slider.get())

        if self.bg_thresolds is not None:

            b,g,r = self.calculate_bg_thresold()
            self.display_image = background_supression_display(self.display_image, (b,g,r))
        
        brightness = self.brigthness_slider.get()
        self.display_image = cv2.convertScaleAbs(self.display_image, 1 , brightness)
        self.display_image_as_label()
        
        if self.ROIs is not None:
            self.update_ROI_data()
        

    def update_ROI_data(self):
        clip_thresold = self.clip_slider.get()

        if self.bg_thresolds is not None:
            b,g,r = self.calculate_bg_thresold()
            ROIs_data = process_ROIs(self.raw_image, self.ROIs, (b,g,r), clip_thresold)
        else:
            ROIs_data = process_ROIs(self.raw_image, self.ROIs, None, clip_thresold)

        self.data_array[self.current_image - 1] = ROIs_data
        self.update_ROIs_label(ROIs_data)
        self.matplotlib_plot(np.array([ROIs_data]))

    
    def display_image_as_label(self):
        
        if self.image_as_label is not None:
            self.image_as_label.destroy()

        image = cv2_draw_ROIs(self.display_image, bg_ROI = self.bg_ROI, ROIs= self.ROIs, colors=self.bgr_color_list)
        color_convert = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.PIL_image = PIL.Image.fromarray(color_convert) 
         
        self.display = ctk.CTkImage(self.PIL_image, size=(800,600))
        self.image_as_label = ctk.CTkLabel(self.Tk_root, image=self.display, text="")
        self.image_as_label.place(x=0,y=0)


    def cv2_bg_selector(self):
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 800, 600)
        self.bg_ROI = cv2.selectROI("Image", self.raw_image)
        self.bg_thresolds = get_background_values(self.raw_image, self.bg_ROI)
        self.clear_plot_and_data_array()
        self.draw_image()

    def calculate_bg_thresold(self):
            x_std_b = self.b_slider.get()
            x_std_g = self.g_slider.get()
            x_std_r = self.r_slider.get()
            b = self.bg_thresolds[0][0] + self.bg_thresolds[0][1] * x_std_b
            g = self.bg_thresolds[1][0] + self.bg_thresolds[1][1] * x_std_g
            r = self.bg_thresolds[2][0] + self.bg_thresolds[2][1] * x_std_r
            return b,g,r
    
    def cv2_ROIs_selector(self):
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 800, 600)
        self.ROIs = cv2.selectROIs("Image", self.display_image)
        cv2.destroyWindow("Image")
        self.clear_plot_and_data_array()
        self.draw_image()

    ##################################
    #### Data Processing and plot ####
    ##################################

    def process_folder(self):
        return
   
    def matplotlib_make_plot(self):
        self.fig, self.ax = plt.subplots()
        self.fig.set_dpi(100) # i inch = 100px
        self.fig.set_size_inches(8,3)
        self.ax.set_position([0.08, 0.15, 0.8, 0.8])
        self.ax.spines[['right', 'top']].set_visible(False)
        self.ax.set_xlabel("Time(s)")
        self.ax.set_ylabel("avg px intensity")
        self.ax.set_xlim(left=0, right= self.total_images*self.second_per_image)
    
    def matplotlib_plot(self, data_array=None):

        if data_array is not None:
            shape = data_array.shape
            if shape[0] == 1:
                x = self.second_per_image * (self.current_image - 1)
            else:
                x = [i*self.second_per_image for i in range(1, shape[0]+1)]
            
            for i in range(shape[1]):
                y = data_array[0:, i , 1 , 0]
                e = data_array[0:, i , 1 , 1]
                self.ax.errorbar(x=x, y=y, yerr=e, linestyle='None', marker='.', elinewidth=1, color=self.color_list[i], label=f"Box {i+1}")
                #plt_smooth(ax, y, x, 10, color=color_list[i])
            
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())
        
        canvas = FigureCanvasTkAgg(self.fig, master=self.Tk_root)
        canvas.draw()
        canvas.get_tk_widget().place(x=10,y=660)

    #### Make Video ####
    @profile
    def draw_PIL_image(self, img_number):
        x=self.Tk_root.winfo_rootx()
        y=self.Tk_root.winfo_rooty()
        x1=x+800
        y1=y+980
        PIL.ImageGrab.grab().crop((x,y,x1,y1)).save(f"D:\\tmp\\{img_number}.jpg")






if __name__ == "__main__": 
    
    path = "D:\\microscope_data\\time-lapse\\231030_154809\\"
    Tk_root = ctk.CTk()
    Tk_root.geometry("1050x980+0+0")
    interface = Interface(Tk_root, path)    
    Tk_root.mainloop()

