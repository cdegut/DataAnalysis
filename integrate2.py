#!/usr/bin/python3

import multiprocessing as mp
import sys
import time
import tkinter
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
from scipy.signal import argrelextrema

########### Options ######################
##########################################

Def_Image_Color_Sacling = 25
timePoint = [0, 20, 40, 75]
BackgroundPercent = 0.05
presetRectangle = 0, 120, 0, 120  # x1 x2 y1 y2
minPeakSpacing = 15


########## Start shared arrays and queue
integration = mp.Queue()
boundaries = mp.Queue()
FinalData = mp.Queue()




################## Load image
try:
    im = cv2.imread(str(sys.argv[1]), -1)
except:
    im = cv2.imread('', -1)  # test file
try:
    im.shape
except:
    print('no image loaded')


######################################
# Integrate row and return multiple data
#######################################

def rowIntegrate(im, x1, x2, y1, y2):
    rowIntegration = []
    for row in range(y1, y2):
        rowPX = []
        for column in range(x1, x2):
            rowPX.append(im[row, column])
        rowSTD = np.std(rowPX)
        rowValue = np.sum(rowPX)
        rowData = [row, rowSTD, rowValue]
        rowIntegration.append(rowData)

    return rowIntegration


##########################################
## integrate column and return integration
##########################################

def columnIntegrate(im, x1, x2, y1, y2):
    integration = []
    for column in range(x1, x2):
        columnValue = 0
        for row in range(y1, y2):
            columnValue = columnValue + im[row, column]
        integration.append(columnValue)
    return integration


####################################
### get rectangle selection position
####################################
def line_select_callback(eclick, erelease, ):
    global x1, y1, x2, y2
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)




###########################
## Key detection for viewer
###########################
def key_detect(event):
    pass


########## Main Integration process #################
#####################################################

def DataIntegrate(im, x1, x2, y1, y2, minPeakSpacing, BackgroundMethod, BackgroundMultiplier, Backgrounda, Backgroundb,
                  addBox):
    ######
    # Selection data
    width = x2 - x1
    height = y2 - y1
    #########################################
    # Data Analysis

    # Generate data about each row
    allRow = rowIntegrate(im, x1, x2, y1, y2)  # integrate all row and get STD for each
    STDsorted = sorted(allRow, key=lambda x: x[1])  # sort array by STD

    allColumn = columnIntegrate(im, x1, x2, y1, y2)  # integrate all collumn

    #########Background for column every column based on n% lowest STD row
    columnValues = []
    columnBackground = []
    n = round(len(STDsorted) * BackgroundPercent)

    for column in range(x1, x2):
        columnValues.clear()
        for row in range(0, n, 1):  # take only 5# 0% with lowest STD
            columnValues.append(im[STDsorted[row][0], column])
        background = np.mean(columnValues) * height
        columnBackground.append(background)


    ########Generate new column with background deleted
    if BackgroundMethod == 1:
        noBackground = []
        for x, column in enumerate(allColumn):
            noBackground.append(allColumn[x] - columnBackground[x])

    elif BackgroundMethod == 2:  # background deleted as a simple mean
        noBackground = []
        meanBack = int(np.mean(columnBackground))
        for x, column in enumerate(allColumn):
            noBackground.append(allColumn[x] - meanBack)

    elif BackgroundMethod == 3:  # background deleted as a simple mean
        noBackground = []
        meanBack = int(np.mean(columnBackground))
        for x, column in enumerate(allColumn):
            noBackground.append(allColumn[x] - Backgrounda * x - meanBack * Backgroundb)

    else:
        noBackground = allColumn

    #######Define local minimas of the function
    minimas = [0]  # get sure ther is at least a minima at the beginin
    for xi in argrelextrema(np.array(noBackground), np.less)[0]:
        if noBackground[xi] <= columnBackground[xi] * BackgroundMultiplier:
            minimas.append(xi)
    minimas.append(width)  # get sure ther is at least a minima at the end

    ###### Cut the array between mimimas for integrtion
    """
    boundary = []
    boxes = []
    for lst, xi in enumerate(minimas):
        try:
            if minimas[lst + 1] - xi > minPeakSpacing:
                box = [xi, minimas[lst + 1]]
                boxes.append(box)
                boundary.append(xi)
                boundary.append(minimas[lst + 1])
        except:
            pass


    ##Add extra boxese for weak signal
    if addBox == 1:
        try:
            if boxes[-1][-1] + minPeakSpacing < width:
                box = [boxes[-1][1], width]
                boxes.append(box)
                boundary.append(width)
        except:
            pass
        try:
            if boxes[0][0] - minPeakSpacing > 0:
                box = [0, boxes[0][0]]
                boxes.append(box)
                boundary.append(0)
        except:
            pass """

# Generate uniform boxes
    boundary = []
    boundary.append(0)
    boxes = []
    boxwidth = minPeakSpacing

    try:
        int(width / boxwidth)
    except:
        boxwidth = 15

    for i in range(0 , int(width / boxwidth)) :
        b1 = int(i * boxwidth)
        b2 = int(b1 + boxwidth)
        boxes.append([b1,b2])
        boundary.append(b1)



    # integrate by peaks
    peaks = []
    #boxes.sort(key=lambda x: x[0])  # be sure boxes are in correct order befor integration

    for peak in boxes:
        peakIntegrale = []
        for n in range(peak[0], peak[1]):
            peakIntegrale.append(noBackground[n])
            peakmidlle = int((peak[1] + peak[0]) / 2)
        peakdata = [peakmidlle, int(sum(peakIntegrale))]
        peaks.append(peakdata)

    return (minimas, columnBackground, allColumn, boundary, noBackground, peaks)


########## Integration ploter ###########
########################################

def plotIntegration(minimas, columnBackground, allColumn, boundary, noBackground, peaks):
    plt.clf()
    # plt.show(block=False)
    ##================= Generate plots
    # Row Data

    plt.subplot(311)
    #for xi in (minimas): plots minimas of the function
    #   plt.axvline(x=xi, color='lightgrey')
    plt.plot(columnBackground)
    plt.plot(allColumn)
    # Filtered
    plt.subplot(312)
    for xi in (boundary):
        plt.axvline(x=xi, color='red')  # plot boudary lines

    plt.plot(noBackground, color='green')  # plot backgrounde filtered data

    plt.axhline(color='black')

    # Peaks histogram, use timepoint if not out of range
    plt.subplot(313)
    if len(peaks) <= len(timePoint):
        for t, peak in enumerate(peaks):
            plt.bar(timePoint[t], peak[1], width=10, color='green', label=peak[1])
    else:
        # if list out of range just plot normaly
        for t, peak in enumerate(peaks):
            plt.bar(t, peak[1], width=0.8, color='green')


######### Image Viewer #################
########################################
def ImageVis(im, selection, options ):
    ### Image handeling

    fig, current_ax = plt.subplots()  # make a new plotting range
    while True:


        plt.plot()

        key_detect.RS = RectangleSelector(current_ax, line_select_callback,
                                          drawtype='box', useblit=True,
                                          button=[1, 3],  # don't use middle button
                                          minspanx=5, minspany=5,
                                          spancoords='pixels',
                                          interactive=True)

        Image_Color_Sacling = options[6] * 1000
        plt.imshow(im, cmap='hot', vmax=Image_Color_Sacling)
        plt.connect('key_press_event', key_detect)


        #key_detect.RS.set_active(True)


        #plt.colorbar()
        plt.draw()
        while plt.waitforbuttonpress(timeout=-1) == False or None:
            time.sleep(0.1)

        try:
            selection[0] = x1
            selection[1] = x2
            selection[2] = y1
            selection[3] = y2
        except:
            selection[0] = presetRectangle[0]
            selection[1] = presetRectangle[1]
            selection[2] = presetRectangle[2]
            selection[3] = presetRectangle[3]



"""def click(event, x, y, flags, param):
    global refPt
    # grab references to the global variables
    global refPt

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        #print(refPt)

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))

        # draw a rectangle around the region of interest
        cv2.rectangle(im, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", im)

def ImageVis(im, selection):
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click)
    while True:
        cv2.imshow("image", im)
        key = cv2.waitKey(1) & 0xFF
"""
def ImageVisZoom(im, selection, options, boundaries):

    ### Image handeling

    while True:
        boundary = boundaries.get()

        Image_Color_Sacling = options[6] * 1000

        plt.imshow(im, cmap='hot', vmax=Image_Color_Sacling)
        plt.xlim(selection[0],selection[1])
        plt.ylim(selection[3],selection[2])


        for line in boundary[0]:
            plt.axvline(line + selection[0])
        #plt.colorbar()
        plt.draw()
        plt.pause(0.01)
        plt.clf()


    ## Integrator as a process worker

def IntegratorWorker(options, integration, selection, boundaries, FinalData ):
    while True:
        minimas, columnBackground, allColumn, boundary, noBackground, peaks = DataIntegrate(im, selection[0],
                                                                                            selection[1],
                                                                                            selection[2],
                                                                                            selection[3], options[0],
                                                                                            options[1], options[2],
                                                                                            options[3], options[4],
                                                                                            options[5])
        integration.put([minimas, columnBackground, allColumn, boundary, noBackground, peaks])
        boundaries.put([boundary])
        FinalData.put([peaks])

        time.sleep(0.4)

def getdata():

    print(dataarray)

## Option Window ##
def Options(options, FinalData):

    global dataarray
    dataarray = []

    master = tkinter.Tk()
    scale = tkinter.Scale(master, from_=1, to=50)
    scale.set(25)

    p = tkinter.Scale(master, from_=5, to=25)
    p.set(15)
    pL = tkinter.Label(master, text='Peak distance')

    b = tkinter.Scale(master, from_=0, to=5, resolution=0.1)
    bL = tkinter.Label(master, text='Backgroud multiplier')
    b.set(2)

    bML = tkinter.Label(master, text='Backgroud Method')
    ba = tkinter.Scale(master, from_=-10, to=10, resolution=0.1)
    ba.set(1)
    baL = tkinter.Label(master, text='a value')
    bb = tkinter.Scale(master, from_=0, to=3, resolution=0.05)
    bb.set(1)
    bbL = tkinter.Label(master, text='b value')

    MODES = [("None", 0), ("per Column", 1), ("overall", 2), ("parametric", 3), ]
    v = tkinter.IntVar()
    v.set(1)  # initialize

    addBox = tkinter.IntVar()
    c = tkinter.Checkbutton(master, text="Add boudaries", variable=addBox)
    c.select()


    B = tkinter.Button(master, text="GetData", command=getdata)

    scale.pack()
    pL.pack()
    p.pack()
    bL.pack()
    b.pack()
    c.pack()
    bML.pack()
    B.pack()

    for text, mode in MODES:
        bM = tkinter.Radiobutton(master, text=text, variable=v, value=mode)
        bM.pack()

    baL.pack()
    ba.pack()
    bbL.pack()
    bb.pack()
    while True:
        master.update()
        dataarray = FinalData.get()
        options[0] = p.get()
        options[1] = v.get()
        options[2] = b.get()
        options[3] = ba.get()
        options[4] = bb.get()
        options[5] = addBox.get()
        options[6] = scale.get()
        time.sleep(0.001)

def main(im):

    # import images -1 is 16bit greyscale flag
    options = mp.Array('f', 7, )
    selection = mp.Array('i', 5, )
    ## Default Value
    selection[0] = presetRectangle[0]
    selection[1] = presetRectangle[1]
    selection[2] = presetRectangle[2]
    selection[3] = presetRectangle[3]
    options[6] = Def_Image_Color_Sacling  # Load default because option selector is not up before the viewer
    # x1, x2, y1, y2 = presetRectangle  # intitiate rectangle if nothing selected


    optionsSelector = mp.Process(target=Options, args=(options, FinalData))
    optionsSelector.start()

    viewer = mp.Process(target=ImageVis, args=(im, selection, options,))
    viewer.start()

    integrator = mp.Process(target=IntegratorWorker, args=(options, integration, selection, boundaries, FinalData ))
    integrator.start()

    zoom = mp.Process(target=ImageVisZoom, args=(im, selection, options, boundaries ))
    zoom.start()

    while optionsSelector.is_alive():

        minimas, columnBackground, allColumn, boundary, noBackground, peaks = integration.get()
        plotIntegration(minimas, columnBackground, allColumn, boundary, noBackground, peaks)
        plt.draw()
        plt.pause(0.05)
        if selection[4] == 1:
            textBox = tkinter.Tk()
            T = tkinter.Text(master=textBox)
            T.pack()

            selection[4] = 0
            for peak in peaks:
                print(peak[1])
                T.insert(tkinter.END, peak[1])
                T.insert(tkinter.END, "\n")

    plt.close()
    integrator.terminate()
    optionsSelector.terminate()
    viewer.terminate()
    zoom.terminate()

if __name__ == '__main__':
    main(im)
