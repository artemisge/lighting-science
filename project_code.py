from DMXEnttecPro import Controller
from luxpy.toolboxes import spectro as sp# import jeti
import luxpy as lx
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import math
#from ctypes import *

def changeColors(*args):
    # R,G,B,A,WW,UV
    dmx.set_channel(1, args[0])
    dmx.set_channel(2, args[1])
    dmx.set_channel(3, args[2])
    dmx.set_channel(4, args[3])
    dmx.set_channel(5, args[4])
    dmx.set_channel(6, args[5])
    dmx.submit()

# doesn't work. path needs to be replaced
#dmx = Controller("/dev/ttyUSB0")  # Typical of Linux
dmx = Controller("COM7")
# try sudo -i, ls /dev before after, udevadm monitor
# and plug in device and then sudo chmod a+rw /dev/ttyUSB0

# array = [0,255,0,0,0,0]
# changeColors(*array)

# JETI INITIALIZATION
sp.init('jeti')

# global var
step = 85#15
wavelengths_array = []
wl_saved = False
# TASK 2
# Measure spectra of all channels (maybe not UV)
def auto_measure(UV_on):
    array_measured_spectra = [] # in the end it will be [5 or 6]x[255/step]
    global wavelengths_array
    if UV_on:
        end = 6 # UV on
    else:
        end = 2 # UV off
    tmp1 = True
    tmp2= True
    for i in range(end):
        channel_array = []
        # step = 15 -> 17 different measurements
        for j in range(0, 255, step):
            DMX_arguments_tmp_array = [0]*6
            # we will measure every channel, increasing the value (0-255) with an interval.
            # for every channel, make the arguments for the DMX:
            # eg: for i = 0 (testing RED channel), and value j = 155 -> DMX arguments will be: [155, 0, 0, 0, 0, 0]
            # and for i = 3 and j = 255 -> [0, 0, 0, 255, 0, 0]
            DMX_arguments_tmp_array[i] = j
            
            changeColors(*DMX_arguments_tmp_array)
            #print(sp.__file__)
            spd = sp.get_spd(manufacturer='jeti')
            if tmp1:
                print("spectra size " + str(np.array(spd).shape)) 
                tmp1 = False
                wavelengths_array = spd[0]
                #print(wavelengths_array)
            channel_array.append(spd[1])
        if tmp2:
            print("channel array size: " + str(np.array(channel_array).shape))
            tmp2 = False
        array_measured_spectra.append(channel_array)

    return array_measured_spectra

def make_luminance_plots(array_measured_spectra):
    # convert to photometric units using the CIE 1931 2° observer (default):
    array_luminances = []
    array_normalized_luminances = []
    print("len: "+str(len(array_measured_spectra)))
    # for every channel (=5)
    for i in range(len(array_measured_spectra)):
        array_luminance_i = []
        # for every measurement of a driver value in this channel (=255/step)
        for j in range(len(array_measured_spectra[0])):
            tmp = lx.spd_to_power(np.vstack([wavelengths_array, array_measured_spectra[i][j]]), ptype = 'pu')
            array_luminance_i.append(tmp[0][0]) # for some reason spd_to_power returns a 3D array instead of a single value
            # print("illuminance of channel measurement: " + str(np.array(tmp).shape))
        # now we have all luminances for all measurements for different driver values for this specific i-channel. And we'll make a plot

        print(array_luminance_i)
        print("size of illum: " +str(np.array(array_luminance_i).shape))

        x = [i for i in range(0, 255, step)]
        print(x)
        plt.plot(x, array_luminance_i, **{'color': 'lightsteelblue', 'marker': 'o'})
        plt.xlabel('Driver Values')
        plt.ylabel('Luminance')
        plt.title('Luminance for different driver values')
        plt.show() # 1 plot per channel
        array_luminances.append(array_luminance_i)

        # also do the normalized luminance (to the max luminance of each channel)
        array_normalized_luminance_i = lx.spd_normalize(np.array(array_luminance_i), norm_type = 'max', wl=False)
        print(np.array(array_luminance_i).shape)

        array_normalized_luminances.append(array_normalized_luminance_i)
        print("normalized" + str(array_normalized_luminance_i))
        
        plt.plot(x, array_normalized_luminance_i)
        plt.xlabel('Driver Values')
        plt.ylabel('Normalized Luminance')
        plt.title('Normalized Luminance for different driver values')
        plt.show() # 1 plot per channel
    return array_normalized_luminances

def interpolate_normalized_luminances(array_normalized_luminances):
    x = [i for i in range(0, 255, step)]
    # cs_x -> give x to find y
    # cs_y -> give y to find x
    cs_x = CubicSpline(x, array_normalized_luminances)
    cs_y = CubicSpline(array_normalized_luminances, x)
    return cs_x, cs_y

# testing interpolation - WORKS
# array = [i*i-i for i in range(0, 255, step)] #instead of normalized luminances
# print(array)
# cs_x, cs_y = interpolate_normalized_luminances(array)
# print(cs_x(200), cs_y(40000))

# call function to measure all channels, with UV off, and print result.
# it should be a 5x17 array. (17, because we took intervals of 15 while testing different driver values for each channel: 0-255)

beautiful_thing = array_measured_spectra = auto_measure(False)
#print(beautiful_thing)
print(" SIZEEE :" + str(np.array(beautiful_thing).shape))
n_l = make_luminance_plots(beautiful_thing)
# x,y=interpolate_normalized_luminances(n_l)
# print(x,y)




# ____TASK 3____
def task3():
    # EEW tristimulous values
    Yxy_eew = lx.xyz_to_Yxy(np.array([[100,100,100]]))

    target_luminance = 100
    # find the correct driver values for R,G,B that have luminance = 100 (or as close to 100)
    saved_colors = []
    driver_values = []
    total_measurements = len(array_measured_spectra[0])
    for i in range(3):
        # we only want RGB channels -> (0,1,2)
        for j in range(total_measurements):
            spd = array_measured_spectra[i][j]
            XYZ = lx.spd_to_xyz(spd, cieobs='1964_10', relative=False)
            # check if lumimance matches 100, Y = luminance
            if math.isclose(XYZ[1], 100, abs_tol=0.5):
                # good job :)
                saved_colors.append(XYZ)
                driver_values.append(j*15)
                break
    # our final driver values are:
    print(driver_values)
    changeColors(driver_values, 0, 0, 0)
    # next step is to use the jeti :)
    spd = sp.get_spd(manufacturer = 'jeti') 
    XYZ = lx.spd_to_xyz(spd, cieobs='1964_10', relative=False)
    # find spectral radiance
    sr = lx.spd_to_power(spd, ptype='ru')
    print("S. Radiance: " + sr)
    Yxy = lx.xyz_to_Yxy(XYZ)

    target_x_y = 1/3 # EEW -> 1/3
    print('Target: 100, ' + target_x_y + ", " + target_x_y)
    print('Result: ' + Yxy)

    # plot chromaticity diagram
    axh = lx.plotSL(cspace='Yuv', cieobs='1964_10', show=False, BBL=True, DL=True, diagram_colors=True)
    Y,u,v = lx.xyz_to_Yuv(XYZ)
    lx.plot_color_data(u,v,formatstr='go', axh=axh)

    # Make ellipses
    v_mac = lx.deltaE.get_macadam_ellipse(nsteps=10)
    lx.plotellipse(v_mac, axh=axh, show=True, cspace_out='Yuv', line_style='-', line_color='w', line_width=1.5)

    # Estimate Macadam ellipses:
    v_mac_est = lx.deltaE.get_macadam_ellipse(xy = Yxy[1:], nsteps = 10)

    lx.plotellipse(v_mac_est, axh = axh, show = True, cspace_out = 'Yuv',\
            line_style = '-', line_color ='w', line_width = 1.5,\
            plot_center = True, center_marker = '.', center_color = 'w', center_markersize = 6)