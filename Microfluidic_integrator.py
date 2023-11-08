
import cv2 
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PIL.Image
import customtkinter as ctk
import glob



def background_value_determination(image, ROI, x_std=(2,2,2)):
    background = image[int(ROI[1]):int(ROI[1]+ROI[3]),int(ROI[0]):int(ROI[0]+ROI[2])]
    (back_b_channel, back_g_channel, back_r_channel) = cv2.split(background)

    b_thresold = np.mean(back_b_channel)+np.std(back_b_channel)*x_std[0]
    g_thresold = np.mean(back_g_channel)+np.std(back_g_channel)*x_std[1]
    r_thresold = np.mean(back_r_channel)+np.std(back_r_channel)*x_std[2]

    return ( b_thresold, g_thresold, r_thresold)


def background_supression(image, thresolds, mode=2):
    
    (b_channel, g_channel, r_channel) = cv2.split(image)

    if mode == 1:
        b, b_thresold_cut = cv2.threshold(b_channel,thresolds[0],255,cv2.THRESH_TOZERO)
        g, g_thresold_cut = cv2.threshold(g_channel,thresolds[1],255,cv2.THRESH_TOZERO)
        r, r_thresold_cut = cv2.threshold(r_channel,thresolds[2],255,cv2.THRESH_TOZERO)
    
    if mode == 2:
        b_thresold_cut = b_channel.astype(int) - int(thresolds[0])
        b_thresold_cut[b_thresold_cut < 0 ] = 0

        g_thresold_cut = g_channel.astype(int) - int(thresolds[1])
        g_thresold_cut[g_thresold_cut < 0 ] = 0
        r_thresold_cut = r_channel.astype(int) - int(thresolds[2])
        r_thresold_cut[r_thresold_cut < 0 ] = 0

    return cv2.merge([b_thresold_cut.astype("uint8"), g_thresold_cut.astype("uint8") , r_thresold_cut.astype("uint8")])

def highlight_supression(image, thresold=200):
    (b_channel, g_channel, r_channel) = cv2.split(image)
    mask = np.add(b_channel, g_channel, r_channel)
    mask[mask < thresold] = 1
    mask[mask >= thresold] = 0
    clipped = cv2.merge((np.multiply(b_channel,mask), np.multiply(g_channel,mask), np.multiply(r_channel,mask)))

    return clipped

def cv2_draw_ROIs(image, bg_ROI=None, ROIs=None):
    if ROIs is not None:
        font = cv2.FONT_HERSHEY_DUPLEX
        i = 1
        for ROI in ROIs:
            image = cv2.rectangle( image, (ROI[0], ROI[1]), (ROI[0]+ROI[2], ROI[1]+ROI[3]), (0,0,255), 4)
            image = cv2.putText(image, str(i), (ROI[0]+ROI[2], ROI[1]), font, fontScale=4, color=(0,0,255), thickness=6)
            i = i+1
    if bg_ROI:
        image = cv2.rectangle( image, (bg_ROI[0], bg_ROI[1]), (bg_ROI[0]+bg_ROI[2], bg_ROI[1]+bg_ROI[3]), (0,255,255), 4)
        image = cv2.putText(image, "Bg", (bg_ROI[0]+bg_ROI[2], bg_ROI[1]), font, fontScale=4, color=(0,255,255), thickness=6 )
    
    return image

def process_ROIs(image, ROIs):
    
    ROIs_data = []
    for ROI in ROIs:
        ROI_crop = image[int(ROI[1]):int(ROI[1]+ROI[3]),int(ROI[0]):int(ROI[0]+ROI[2])]
        (b_channel, g_channel, r_channel) = cv2.split(ROI_crop)

        b_channel[b_channel != 0]
        g_channel[g_channel != 0]
        r_channel[r_channel != 0]

        b = np.mean(b_channel)
        b_std = np.std(b_channel)
        g = np.mean(g_channel)
        g_std = np.std(g_channel)
        r = np.mean(r_channel)
        r_std = np.std(r_channel)

        
        l = [(b, b_std),(g, g_std), (r, r_std)]
        ROIs_data.append(l)
    
    return ROIs_data
    
def folder_process(thresolds, clip, ROIs, path="D:\\microscope_data\\time-lapse\\231030_154809\\"):
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 800, 600)

    data = []

    for image in glob.glob(f"{path}\\*.png"):
        current_image = cv2.imread(image)
       
        if clip:
            current_image = highlight_supression(current_image)
        
        if thresolds:
            current_image = background_supression(current_image, thresolds)
        
        ROIs_data = process_ROIs(current_image, ROIs)
       
        data.append(ROIs_data)
        

        current_image = cv2_draw_ROIs(current_image, bg_ROI = None, ROIs=ROIs)
        cv2.imshow("Image", current_image)
        cv2.waitKey(10)
    data_array = np.array(data)
    print(data_array)
    cv2.destroyWindow("Image")
    return data_array
        
        


class Interface(ctk.CTkFrame):
    def __init__(self, Tk_root, path):
        ctk.CTkFrame.__init__(self, Tk_root)
        self.Tk_root = Tk_root
        self.bg_ROI = None
        self.redraw_job = None

        self.default_bgr_sliders = (2,0,2)
        self.default_clip_slider = 200

        self.images_list = glob.glob(f"{path}\\*.png")
        self.image = cv2.imread(self.images_list[0])


        self.ROIs = []

        self.init_window()

    def init_window(self):
        self.bgr_sliders_draw((820, 10))
        self.clip_slider_draw((820, 190))
        self.images_slider_draw((400,600), 750)
        button_position = (840, 280)

        #redraw_button = ctk.CTkButton(master=self.Tk_root, text="Redraw", command=self.display_image)
        #redraw_button.place(x=650, y=620, anchor=ctk.N)
        default_button = ctk.CTkButton(master=self.Tk_root, text="Default",width=180, command=self.default)
        default_button.place(x=button_position[0], y=button_position[1])
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
        self.ROIs_label.place(x=820, y = 540)

        self.display_image()

    def images_slider_draw(self, position, lengh):
        totalimages = len(self.images_list)
        self.images_slider = ctk.CTkSlider(master = self.Tk_root, from_=0, to=totalimages-1, number_of_steps=totalimages-1, width=lengh, command=self.load_image, orientation=ctk.HORIZONTAL)
        self.images_slider.place(x=position[0], y = position[1]+40, anchor=ctk.N)
        self.images_slider.set(0)
        self.image_label = ctk.CTkLabel(master = self.Tk_root, text = "Image #: 1")
        self.image_label.place(x=position[0]-380, y = position[1]) 

    def bgr_sliders_draw(self, position):
        
        self.b_slider = ctk.CTkSlider(master = self.Tk_root, from_=-10, to=10, number_of_steps=20,  command=self.redraw, orientation=ctk.HORIZONTAL)
        self.g_slider = ctk.CTkSlider(master = self.Tk_root, from_=-10, to=10, number_of_steps=20, command=self.redraw, orientation=ctk.HORIZONTAL)
        self.r_slider = ctk.CTkSlider(master = self.Tk_root, from_=-10, to=10,  number_of_steps=20, command=self.redraw, orientation=ctk.HORIZONTAL)
        
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
    
    def default(self):
        self.b_slider.set(self.default_bgr_sliders[0])
        self.g_slider.set(self.default_bgr_sliders[1])
        self.r_slider.set(self.default_bgr_sliders[2])
        self.clip_slider.set(self.default_clip_slider)
    
        self.display_image()
    
    def clear_bg(self):
        self.bg_ROI = None
        self.redraw()

    def clip_slider_draw(self, position):
        self.clip_slider = ctk.CTkSlider(master = self.Tk_root, from_=0, to=400, number_of_steps=40, command=self.redraw,  orientation=ctk.HORIZONTAL)
        self.clip_label = ctk.CTkLabel(master = self.Tk_root, text = "Highlight clipping level:")
        self.clip_label.place(x=position[0], y = position[1])
        self.clip_slider.place(x=position[0]+10, y = position[1]+30)
        self.clip_slider.set(self.default_clip_slider)

    def label_update(self):
        self.b_label.configure(text= f"Blue: {self.b_slider.get()}")
        self.g_label.configure(text= f"Green: {self.g_slider.get()}")
        self.r_label.configure(text= f"Red: {self.r_slider.get()}")
        self.clip_label.configure(text= f"Highlight clipping level: {self.clip_slider.get()}")
        self.after(100, self.label_update)
    

    def load_image(self, x):
        i = int(x)
        self.image = cv2.imread(self.images_list[i])
        self.image_label.configure(text=f"Image #: {i+1}")
        self.redraw()
    
    def cv2_bg_selector(self):
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 800, 600)
        self.bg_ROI = cv2.selectROI("Image", self.image)
        cv2.destroyWindow("Image")
        self.display_image()
    
    def cv2_ROIs_selector(self):
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 800, 600)
        image = self.process_display_image()
        self.ROIs = cv2.selectROIs("Image", image)
        cv2.destroyWindow("Image")
        self.display_image()


    def redraw(self, x=None):
        
        if self.redraw_job:
           
            self.Tk_root.after_cancel(self.redraw_job)
        
        self.redraw_job = self.after(50, self.display_image)
        

    def display_image(self):
        image = self.process_display_image()
        
        if self.ROIs is not None:
            self.update_ROIs_label(image)

        self.label_display_image(image)

    
    def update_ROIs_label(self, image):
        
        ROIs_data = process_ROIs(image, self.ROIs)
        text = "ROIs integration data:\n"
        i = 1
        for ROI_data in ROIs_data:
            label = f"\n #{i}  B:{round(ROI_data[0][0],2)}  G:{round(ROI_data[1][0],2)}  R:{round(ROI_data[2][0],2)}"
            text = text + label
            i = i +1

        self.ROIs_label.configure(text = text)

    def process_display_image(self):

        image = self.image
        image = highlight_supression(image, self.clip_slider.get())

        if self.bg_ROI:
            thresolds = background_value_determination(image, self.bg_ROI, (self.b_slider.get(),self.g_slider.get(),self.r_slider.get()))
            image = background_supression(image, thresolds)
        
        return image

    
    def label_display_image(self, image):
        image = cv2_draw_ROIs(image, bg_ROI = self.bg_ROI, ROIs= self.ROIs)
        color_convert = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        PIL_image = PIL.Image.fromarray(color_convert) 
        display = ctk.CTkImage(PIL_image, size=(800,600))
        image_label = ctk.CTkLabel(self.Tk_root, image=display, text="")
        image_label.place(x=0,y=0)


    def process_folder(self):
        bg_values =(self.b_slider.get(),self.g_slider.get(),self.r_slider.get())
        clip = self.clip_slider.get()
        
        thresolds = None
        if self.bg_ROI:
           thresolds =  background_value_determination(self.image, self.bg_ROI, bg_values)

        data_array = folder_process(thresolds, clip, self.ROIs, path="D:\\microscope_data\\time-lapse\\231030_154809\\")
        
        self.matplotlib_plot(data_array)
    
    def matplotlib_plot(self, data_array):

        fig, ax = plt.subplots()
        fig.set_size_inches(8,3)
        fig.set_dpi(100)
        #fig.margins(x=0,y=0)
        #fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        #fig.axes((0.0, 0.0, 0.0, 0))
        fig.gca().set_position([0.1, 0.1, 0.9, 0.9])

        shape = data_array.shape
        x = range(1, shape[0]+1)

        color_list = ("firebrick", "olivedrab", "steelblue", "dimgrey")
        for i in range(shape[1]):
            y = data_array[0:, i , 1 , 0]
            e = data_array[0:, i , 1 , 1]
            ax.errorbar(x=x, y=y, yerr=e, linestyle='None', marker='.', elinewidth=1, color=color_list[i], label=f"Box {i+1}")
            #plt_smooth(ax, y, x, 10, color=color_list[i])
            ax.legend()
        
        canvas = FigureCanvasTkAgg(fig, master=self.Tk_root)
        canvas.draw()
        canvas.get_tk_widget().place(x=10,y=660)





if __name__ == "__main__": 
    
    path = "D:\\microscope_data\\time-lapse\\231030_154809\\"
    Tk_root = ctk.CTk()
    Tk_root.geometry("1050x960")
    interface = Interface(Tk_root, path)       
    Tk_root.mainloop()

