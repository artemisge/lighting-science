from DMXEnttecPro import Controller
from luxpy.toolboxes import spectro as sp# import jeti
import luxpy as lx
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import math

# _______________________________________________________________________________
# _________________________________TASK 1________________________________________
# _______________________________________________________________________________


# takes a list of arguments that signify driver values and activates the LED lamp with this specific color
def changeColors(*args):
    # R,G,B,A,WW,UV
    dmx.set_channel(1, args[0])
    dmx.set_channel(2, args[1])
    dmx.set_channel(3, args[2])
    dmx.set_channel(4, args[3])
    dmx.set_channel(5, args[4])
    dmx.set_channel(6, args[5])
    dmx.submit()


# _______________________________________________________________________________
# _________________________________TASK 2________________________________________
# _______________________________________________________________________________


step = 51 #15 # integral divisions of 255. The more the better the results, but more time costly
wavelengths_array = [] # store the wavelengths, instead of having them in the spd arrays
wl_saved = False # to save the wavelengths only one time, and not everytime

# Measure spectra of all channels (maybe not UV)
def auto_measure(UV_on):
    array_measured_spectra = [] # in the end it will be [5 or 6]x[255/step]
    global wavelengths_array # global, in order to keep the value even after this function is executed

    if UV_on:
        end = 6 # UV on
    else:
        end = 1 # UV off # TODO: change it back to '5', when everythign is done

    # For every channel:
    for i in range(end):
        channel_array = [] # array to keep all the spds of this particular channel
        # For each measurement: (#measurements=255/step)
        for j in range(0, 255, step):
            # initialize all the drivers values to zero -> [0,0,0,0,0,0].
            DMX_arguments_tmp_array = [0]*6
            # we will measure every channel, increasing the value (0-255) with an interval/step.
            # for every channel, make the arguments for the DMX:
            # eg: for i = 0 (testing RED channel), and value j = 155 -> DMX arguments will be: [155, 0, 0, 0, 0, 0]
            # and for i = 3 and j = 255 -> [0, 0, 0, 255, 0, 0]
            DMX_arguments_tmp_array[i] = j
            changeColors(*DMX_arguments_tmp_array)

            # after changing the lamp colors, measure the spd:
            spd = sp.get_spd(manufacturer='jeti')

            # HAPPENS ONLY ONCE: saves the wavelengths that are part of the spd (in luxpy) as an array, for practical purposes
            if not wl_saved:
                wl_saved = True
                wavelengths_array = spd[0] # spd[0] carries the wavelengths

            # append the 1D array of the spd of this particular driver value and channel
            channel_array.append(spd[1])
        # append the 2D array of all spds of this particular channel
        array_measured_spectra.append(channel_array)

    # return all measured spds of all channels -> [5, #measurements, 471] is the size of this array (471: defferent wavelengths spanning across the visible spectrum)
    return array_measured_spectra

# takes as an arguments the array that was calculated in the previous function: all the spds for every channel
# and returns the normalized luminances array of size: [#channels, #measurements]
def make_luminance_plots(array_measured_spectra):
    # convert to photometric units using the CIE 1931 2° observer (default):
    array_luminances = []
    array_normalized_luminances = []
    # print("len: "+str(len(array_measured_spectra)))

    # For every channel (=5):
    for i in range(len(array_measured_spectra)):
        array_luminance_i = []
        # For every measurement of a driver value in this channel (=255/step):
        for j in range(len(array_measured_spectra[0])):
            tmp = lx.spd_to_power(np.vstack([wavelengths_array, array_measured_spectra[i][j]]), ptype = 'pu')
            array_luminance_i.append(tmp[0][0]) # for some reason spd_to_power returns a 3D array instead of a single value, so we get the single value by doing this -> [0][0]
            # print("illuminance of channel measurement: " + str(np.array(tmp).shape))

        # now we have all luminances for all measurements for different driver values for this specific i-channel. And we'll make a plot
        x = [i for i in range(0, 255, step)]
        plt.plot(x, array_luminance_i, **{'color': 'lightsteelblue', 'marker': 'o'})
        plt.xlabel('Driver Values')
        plt.ylabel('Luminance')
        plt.title('Luminance for different driver values')
        plt.show() # 1 plot per channel
        array_luminances.append(array_luminance_i)

        # also do the normalized luminance (to the max luminance of each channel)
        array_normalized_luminance_i = lx.spd_normalize(np.copy(array_luminance_i), wl=False) # TODO: FIX!!!!! 'max' is not working! ASK WHY!!!!
        #print(array_luminance_i)
        #print(array_normalized_luminance_i)

        array_normalized_luminances.append(array_normalized_luminance_i)
        
        plt.plot(x, array_normalized_luminance_i, **{'color': 'lightsteelblue', 'marker': 'o'})
        plt.xlabel('Driver Values')
        plt.ylabel('Normalized Luminance')
        plt.title('Normalized Luminance for different driver values')
        plt.show() # 1 plot per channel
    print(array_normalized_luminances)
    return array_normalized_luminances

# function that receives an 1D array with normalized luminances of 1 channel and returns 2 
# functions that are the cubic spline interpolation of the normalized luminances.
# First one can take a x value and return the interpolated y value.
# Second one can take a y value and return the interpolated x value.
def interpolate_normalized_luminances(array_normalized_luminances):
    x = [i for i in range(0, 255, step)]
    # cs_x -> give x to find y
    # cs_y -> give y to find x
    cs_x = CubicSpline(x, array_normalized_luminances[0])
    cs_y = CubicSpline(array_normalized_luminances[0], x)
    return cs_x, cs_y

# HOW ABOVE INTERPOLATION WORKS:
# cs_x, cs_y = interpolate_normalized_luminances(normalized_luminances_array)
# print(cs_x(200), cs_y(40000)) -> should print: cs_x(200) ~= 40000, and cs_y(40000) ~= 200

# _______________________________________________________________________________
# _________________________________TASK 3________________________________________
# _______________________________________________________________________________



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

# _______________________________________________________________________________
# _________________________________TASK 4________________________________________
# _______________________________________________________________________________

# TODO

# _______________________________________________________________________________
# ____________________________MAIN SCRIPT________________________________________
# _______________________________________________________________________________

# call function to measure all channels, with UV off, and print result.
# it should be a 5x17 array. (17, because we took intervals of 15 while testing different driver values for each channel: 0-255)

# ___INITIALIZE DMX AND JETI___
#dmx = Controller("/dev/ttyUSB0")  # Typical of Linux
dmx = Controller("COM7") # Typical of Windows
sp.init('jeti')

# LINUX TIP (to find port): 'sudo -i', 'ls /dev' (compare before and after USB device is plugged, to find the correct port). If access is denied do: 'sudo chmod a+rw /dev/ttyUSB0'


array_measured_spectra = auto_measure(False)
normalized_luminances = make_luminance_plots(array_measured_spectra) # size: channels x measurements
# print(" SIZEEE :" + str(np.array(n_l).shape))

x,y = interpolate_normalized_luminances(normalized_luminances)
# print(x(51),y) # to test the interpolation
