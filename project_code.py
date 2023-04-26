from DMXEnttecPro import Controller
from DMXEnttecPro.utils import get_port_by_serial_number, get_port_by_product_id
from luxpy.toolboxes import spectro as sp# import jeti
import luxpy as lx
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import math
from luxpy.toolboxes import spdbuild as spb
import copy

# _______________________________________________________________________
# _____________________________TASK 1____________________________________
# _______________________________________________________________________


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


# _______________________________________________________________________
# _____________________________TASK 2____________________________________
# _______________________________________________________________________

# Measure spectra of all channels (maybe not UV)
def auto_measure(UV_on, step):
    array_measured_spectra = [] # in the end it will be [5 or 6]x[255/step]x[471] (471: wavelength steps, like [450, 455,...678,680..])
    wavelengths_array = [] # stores the wavelengths spectrum (eg [480, 485, 450......675, 680])
    wl_saved = False # boolean variable useful for saving the spectrum of wavelengths only once

    if UV_on:
        end = 6 # UV on
    else:
        end = 5 # UV off # TODO: change it back to '5', when everythign is done

    # For every channel:
    for i in range(end):
        channel_array = [] # array to keep all the spds of this particular channel
        # For each measurement: (#measurements=255/step)

        DMX_arguments_tmp_array = [0]*6
        DMX_arguments_tmp_array[i] = 5
        changeColors(*DMX_arguments_tmp_array)
        # after changing the lamp colors, measure the spd:
        spd = sp.get_spd(manufacturer='jeti')
        channel_array.append(spd[1])

        for j in range(step, 255+1, step):
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
    return array_measured_spectra, wavelengths_array

# since kevin's function isn't working, we make our own, that normalizes the luminances according to the max luminance of each channel. It does the job for ONE channel at a time.
def normalize_max(array_luminance_i):
    normalized_luminances = []
    max = max(array_luminance_i)

    # divide all elements of array with max, to normalize
    normalized_luminances = array_luminance_i / max
    return normalized_luminances


# takes as an arguments the array that was calculated in the previous function: all the spds for every channel AND the array of the wavelengths spectrum AND the measurement step
# and returns the normalized luminances array of size: [#channels, #measurements] AND the non-normalized luminances
def make_luminance_plots(array_measured_spectra, wavelengths_array, step):
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
        x = [i for i in range(0, 255+1, step)]
        x.insert(1,5)#insert 5 dv in 1 index
        print("len x:" + str(len(x)))
        array_luminance_i.insert(0,0)
        # np.insert(array_luminance_i,0,0)
        print(len(array_luminance_i))
        plt.plot(x, array_luminance_i, **{'color': 'lightsteelblue', 'marker': 'o'})
        plt.xlabel('Driver Values')
        plt.ylabel('Luminance')
        plt.title('Luminance for different driver values')
        plt.show() # 1 plot per channel
        array_luminances.append(array_luminance_i)

        # also do the normalized luminance (to the max luminance of each channel)
        array_normalized_luminance_i = normalize_max(array_luminance_i)
        # array_normalized_luminance_i = lx.spd_normalize(np.copy(array_luminance_i), wl=False) # luxpy:'max' is not working!
        #print(array_luminance_i)
        #print(array_normalized_luminance_i)

        array_normalized_luminances.append(array_normalized_luminance_i)
        
        plt.plot(x, array_normalized_luminance_i, **{'color': 'lightsteelblue', 'marker': 'o'})
        plt.xlabel('Driver Values')
        plt.ylabel('Normalized Luminance')
        plt.title('Normalized Luminance for different driver values')
        plt.show() # 1 plot per channel
        # TODO! GIRLS: if u can make the plots into one pretty image.
    # print(array_normalized_luminances)
    return array_normalized_luminances, array_luminances

# function that receives an 1D array with (normalized) luminances of ONE channel and returns two functions that are the cubic spline interpolation of the luminances and the corresponding driver values.
# First one can take a x value and return the interpolated y value.
# Second one can take a y value and return the interpolated x value.
# Argument 'step' is the step that we increased the driver values to make measurements in task 2.
def interpolate_luminances(array_luminances, step):
    # make sample points of driver values, according to the measurements that we did on task 2.

    # start from 'step', since jeti couldn't measure from 0.
    driver_value = [i for i in range(step, 255+1, step)]
    # Instead of '0', we started the measurements from driver value = 5. So we add it manually.
    driver_value.insert(0,5)
    # And we shouldn't forget the driver value = 0. So again we add it manually
    driver_value.insert(0,0)

    # dv_to_lum -> give driver value to find luminance
    # lum_to_dv -> give luminance to find driver value
    dv_to_lum = []
    lum_to_dv = []

    # For every channel:
    for i in range(len(array_luminances)):
        # In the code above, we inserted manually the driver value = 0, for our 'x' function. Now, we need to do the same for our 'y' function (corresponding to luminances. So we add a luminance = 0 in the beginning of the array)
        np.insert(array_luminances, 0, 0)

        # make the interpolation functions:
        dv_to_lum.append(CubicSpline(driver_value, array_luminances[i]))
        lum_to_dv.append(CubicSpline(array_luminances[i], driver_value))
    
    
    return dv_to_lum, lum_to_dv

# HOW ABOVE INTERPOLATION WORKS, example:
# cs_x, cs_y = interpolate_normalized_luminances(normalized_luminances_array)
# print(cs_x(200), cs_y(40000)) -> should print (example numbers): cs_x(200) ~= 40000, and cs_y(40000) ~= 200

# _______________________________________________________________________
# _____________________________TASK 3____________________________________
# _______________________________________________________________________

# takes as arguments the measured spectra for all channels and returns the primary spectra of each channel (it chooses the last measurement of each channel, that means the one with the higher driver value), and stacks on the start of the array the wavelengths array to make it compatible with luxpy functions
def get_stacked_spd(array_measured_spectra, wavelengths_array, N):
    spd_p = []
    # For every primary/channel:
    for i in range(N):
        # We know that the last measurement has the higher driver value, so it will be on the last element of the array.
        # A python way to get the last element is: some_list[-1]
        max_spd = array_measured_spectra[i][-1]
        # print("max spd for channel :" + str(i) + str(max_spd))
        spd_p.append(max_spd)
    spd_p.insert(0, wavelengths_array)
    spd_p = np.array(spd_p)
    return spd_p

# saves a numpy array to a file
def save_file_measurements(spd, filename):
    np.save(filename, spd)

# takes a filename that contains a numpy array, as an argument, and returns this array after it loads it using a numpy function 
def read_file_measurements(filename):
    return np.load(filename)

# takes as arguments the interpolation functions that we calculated at task2. (Reminder: driver value to luminance | luminance to driver value, for every channel. So dv_to_lum[0](255) will return the luminance of red channel for driver value of 255.) AND measurement step AND number of channels to mix
def task3(dv_to_lum, lum_to_dv, step, N, array_measured_spectra, wavelengths_array, luminances):
    t_l = 100 # Target Luminance: find the correct driver values for R,G,B that have luminance = 100 (or as close to 100)

    # __________
    # Commented code below is just a test to check if our interpolation function lum_to_div works, and it does :)
    # y = []
    # for i in range(0,1000,5):
    #     y.append(lum_to_dv[1](i))
    # x = [i for i in range(0,1000,5)]
    # plt.clf()
    # plt.plot(x, y, **{'color': 'lightsteelblue', 'marker': 'o'})
    # plt.show()
    # __________

    # EEW tristimulous values are XYZ = [100,100,100]
    # EEW chromaticity coordinates are x,y = [1/3, 1/3]
    # EEW Yxy coordinates are Y,x,y = [100,1/3,1/3]
    Yxy_eew = lx.xyz_to_Yxy(np.array([[t_l,t_l,t_l]]))
    print("Yxy eew: " + str(Yxy_eew))

    # We want the spds of all the primaries, but for every channel/primary,we have many different spds, given a driver value. We want to take the measurement with the higher driver value, which is the max lumimance we can get.
    # We are going to stack all of max-spds of the primaries into the variable below
    spd_p = get_stacked_spd(array_measured_spectra, wavelengths_array, N)
    # print(spd_p)
    # print(spd_p.shape)

    # Now we convert the stacked-spd of the primaries into XYZ and then into Yxy coordinates
    XYZp = lx.spd_to_xyz(spd_p, cieobs='1964_10', relative=False)
    Yxyp = lx.xyz_to_Yxy(XYZp) 
    print("Yxy of primaries: " + str(Yxyp))

    # We need to solve w[] = Cp^-1 * Ct[], where w[]: weights vector, Cp^-1: inverse of primaries Yxy coords, and Ct[]: target color vector Yxy coords.
    weights = spb.colormixer_pinv(Yxy_eew,Yxyp,input_fmt='Yxy')[0] # mixing using Yxy, it returns a 2D array, so we use array[0] to get only one dimension
    print("weights of luminance for channels: " + str(weights))

    # (girls, make sure this is theoritically correct:) weights are on a scale of 0-1, and they correspond to the weighted luminance of each channel. We have made a cubic spline interpolation to find the driver value to a luminance value. So the final driver value for each channel that we are going to mix is: the result of the interpolation function lum_to_dv, after we put as input the weighted luminance of this channel (we are selecting the max luminance out of all measured luminances of the channel).
    driver_values = []
    for i in range(N):
      luminance_of_channel = np.array(luminances[i]).max()
      print("luminance of channel:" + str(luminance_of_channel))

      weighted_luminance_of_channel = luminance_of_channel*weights[i]
      print("weighted luminance of channel: " + str(weighted_luminance_of_channel))
      print(lum_to_dv[i](weighted_luminance_of_channel))

      # since interpolation is going to give float result, we use function floor to take the previous integer.
      arg = np.floor(lum_to_dv[i](weighted_luminance_of_channel)).astype(int)

      # if for any reason driver value is larger thatn 255, keep only 255.
      if arg > 255:
          arg = 255
      driver_values.append(arg)
   
    # our final driver values for all channels are:
    print("driver values: " + str(driver_values))
    
    # padding the arguments with the right amount of zeros in the end to "fill" all the driver values of channels we didn't use. THe number of these channels is 6-N.
    arguments = np.pad(driver_values, (0,6-N), 'constant', constant_values=(0))
    print("arguments: " + str(arguments))
    changeColors(*arguments)

    # next step is to use the jeti and measure the spd of the calculated color, that is trying to match the target color. :)
    spd_mixed_measured = sp.get_spd(manufacturer = 'jeti') 
    XYZ_mixed_measured = lx.spd_to_xyz(spd_mixed_measured, cieobs='1964_10', relative=False)

    # find spectral radiance
    sr_mixed_measured = lx.spd_to_power(spd_mixed_measured, ptype='ru')
    print("S. Radiance of mixed color: " + str(sr_mixed_measured))

    # compare Yxy coordinates of both colors
    Yxy_mixed_measured = lx.xyz_to_Yxy(XYZ_mixed_measured) # mixed color
    print('Target Yxy (Luminance, x, y coordinate): ' + str(Yxy_eew))
    print('Result Yxy (Luminance, x, y coordinate): ' + str(Yxy_mixed_measured))

    # plot chromaticity diagram
    # plotSL plots the spectrum locus
    axh = lx.plotSL(cspace='Yuv', cieobs='1964_10', show=False, BBL=True, DL=True, diagram_colors=True)
    Yuv = lx.xyz_to_Yuv(XYZ_mixed_measured)
    print("Result Yuv: " + str(Yuv))
    lx.plot_color_data(Yuv[0][1], Yuv[0][2], formatstr='go', axh=axh) # Yuv[0][1]: u, Yuv[0][2]: v, again, kevin's function return a 2D array, and we need only 1 dimension, that's why we use Yuv[0].

    # Make ellipses: Estimate n-step MacAdam ellipse at CIE x,y coordinates xy by calculating average inverse covariance ellipse of the k_neighbours closest ellipses.
    v_mac = lx.deltaE.get_macadam_ellipse(nsteps=10)
    lx.plotellipse(v_mac, axh=axh, show=True, cspace_out='Yuv', line_style='-', line_color='w', line_width=1.5)

    # Estimate Macadam ellipses:
    # v_mac_est = lx.deltaE.get_macadam_ellipse(xy = Yxy_mixed_measured[1:], nsteps = 10)

    # lx.plotellipse(v_mac_est, axh = axh, show = True, cspace_out = 'Yuv', line_style = '-', line_color ='w', line_width = 1.5, plot_center = True, center_marker = '.', center_color = 'w', center_markersize = 6)

# xyzt3 = np.atleast_2d((xyzp[p3,:].T@wp3).T) -> array.T : transpose. array@array : matrix multiplication

# _______________________________________________________________________
# _____________________________TASK 4____________________________________
# _______________________________________________________________________


# define function that calculates several objectives at the same time (for speed):
def spd_to_cris(spd):
    Rf,Rg = lx.cri.spd_to_cri(spd, cri_type='ies-tm30',out='Rf,Rg')
    return np.vstack((Rf, Rg))  

def optimize(method):
    return

def task4(lum_to_dv, measurement_step, N, array_measured_spectra, wavelengths_array, luminances):
    # TARGET/EEW info
    spd_p = get_stacked_spd(array_measured_spectra, wavelengths_array, N)
    t_l = 100
    XYZ_eew = np.array([[t_l,t_l,t_l]])
    target = lx.xyz_to_Yxy(np.array([[t_l,t_l,t_l]]))
    
    # optimize parameters
    cieobs = '1964_10'
    obj_fcn = [(spd_to_cris,'Rf','Rg')]
    obj_tar_vals = [(90,110)]
    method = 'Nelder-Mead'

    # start optimization:
    so1 = spb.SpectralOptimizer(target = target, tar_type = 'Yxy', cspace_bwtf = {},
                                nprim = N, wlr = [360,830,1], cieobs = cieobs, 
                                optimizer_type = '3mixer',
                                prim_constructor = None, 
                                prims = spd_p, 
                                obj_fcn = spb.ObjFcns(f=obj_fcn, ft = obj_tar_vals),
                                minimizer = spb.Minimizer(method=method),
                                verbosity = 0)

    # start optimization and request optimized spectra spds and primary fluxes M as output:
    S,M = so1.start(out = 'spds,Ms')
    # Check output agrees with target:
    xyz = lx.spd_to_xyz(S, relative = False, cieobs = cieobs)
    Yxy = lx.xyz_to_Yxy(xyz)
    cct,duv = lx.xyz_to_cct(xyz, cieobs = cieobs, out = 'cct,duv')
    Rf, Rg = spd_to_cris(S)
    print('\nResults (optim,target):')
    print(Rf.shape,Rf)
    print(obj_tar_vals[0][0])
    print("Yxy: ([{:1.0f},{:1.2f},{:1.2f}],[{:1.0f},{:1.2f},{:1.2f}])".format(Yxy[0,0],Yxy[0,1],Yxy[0,2],target[0,0],target[0,1],target[0,2]))
    print("Rf: ({:1.2f},{:1.2f})".format(Rf[0], obj_tar_vals[0][0]))
    print("Rg: ({:1.2f}, {:1.2f})".format(Rg[0], obj_tar_vals[0][1]))
    print("cct(K), duv: ({:1.1f},{:1.4f})".format(cct[0,0], duv[0,0]))

    print('\nFlux ratios of component spectra:', M)

    #plot spd:
    plt.figure()
    lx.SPD(S).plot()

    driver_values = []
    for i in range(N):
    #   print("test" + str(S[i+1]))
      luminance_of_channel = np.array(luminances[i]).max()
    #   luminance_of_channel = lx.spd_to_power(S[i+1], 'pu')  #first element is wavelengths
     # print("luminance of channel:" + str(luminance_of_channel))
      #print("M array: " + str(M))
      weighted_luminance_of_channel = luminance_of_channel*M[0][i] # M for some reason returns a 2D array, so we take the M[0] for only 1 dimension
      #print("weighted luminance: " + str(weighted_luminance_of_channel))
      #print(lum_to_dv[i](weighted_luminance_of_channel))
      arg = np.floor(lum_to_dv[i](weighted_luminance_of_channel)).astype(int)
      if arg > 255:
          arg = 255
      driver_values.append(arg)
    #print("driver values: " + str(driver_values))

    arguments = np.pad(driver_values, (0,6-N), 'constant', constant_values=(0))
    print("arguments: " + str(arguments))

    changeColors(*arguments) # The meaning of this '[0]*(6-N)' is to fill the driver values of the channels we didn't use, with zero.

    # next step is to use the jeti and measure the spd of the calculated color, that is trying to match the target color. :)
    spd_optimized_mixed_measured = sp.get_spd(manufacturer = 'jeti') 
    XYZ_optimized_mixed_measured = lx.spd_to_xyz(spd_optimized_mixed_measured, cieobs='1964_10', relative=False)

    # find spectral radiance
    sr_optimized_mixed_measured = lx.spd_to_power(spd_optimized_mixed_measured, ptype='ru')
    print("S. Radiance of optimized mixed color: " + str(sr_optimized_mixed_measured))


     # plot chromaticity diagram
    axh = lx.plotSL(cspace='Yuv', cieobs='1964_10', show=False, BBL=True, DL=True, diagram_colors=True)
    optimized_Yuv = lx.xyz_to_Yuv(XYZ_optimized_mixed_measured)
    print(optimized_Yuv)
    lx.plot_color_data(optimized_Yuv[0][1], optimized_Yuv[0][2], formatstr='go', axh=axh) # Yuv[0][1]: u, Yuv[0][2]: v, again, kevin's function return a 2D array, and we need only 1 dimension, that's why we use Yuv[0].

    # Make ellipses
    v_mac = lx.deltaE.get_macadam_ellipse(nsteps=10)
    lx.plotellipse(v_mac, axh=axh, show=True, cspace_out='Yuv', line_style='-', line_color='w', line_width=1.5)

    return


# _______________________________________________________________________
# _____________________________MAIN SCRIPT_______________________________
# _______________________________________________________________________

# ___INITIALIZE DMX AND JETI___
#dmx = Controller("/dev/ttyUSB0")  # Typical of Linux
#dmx = Controller("COM7") # Typical of Windows

# can use dmx function to find serial port, it's on the site
# my_port = get_port_by_serial_number('EN055555A') -> doesn't work
my_port = get_port_by_product_id(24577)
dmx = Controller(my_port)

sp.init('jeti')

# changeColors(255,0,0,0,0,0) # just a test

# LINUX TIP (to find port): 'sudo -i', 'ls /dev' (compare before and after USB device is plugged, to find the correct port). If access is denied do: 'sudo chmod a+rw /dev/ttyUSB0'

measurement_step = 51 #  integer divisions of 255. The more the better the results, but more time costly

file_saved = True
if not file_saved:
    array_measured_spectra, wavelengths_array = auto_measure(False, measurement_step)
    save_file_measurements(array_measured_spectra, 'array_measured_spectra.npy')
    save_file_measurements(wavelengths_array, 'wavelengths_array.npy')
else:
    array_measured_spectra = read_file_measurements('array_measured_spectra.npy')
    wavelengths_array = read_file_measurements('wavelengths_array.npy')

normalized_luminances, luminances = make_luminance_plots(copy.deepcopy(array_measured_spectra), wavelengths_array, measurement_step) # size: channels x measurements
# print(" SIZEEE :" + str(np.array(n_l).shape))
print("luminancs: " + str(luminances))

# dv_to_lum_norma, lum_to_dv_norma = interpolate_luminances(normalized_luminances, measurement_step)
dv_to_lum, lum_to_dv = interpolate_luminances(luminances, measurement_step)

N = 4 # number of colors/channels we want to mix
task3(dv_to_lum, lum_to_dv, measurement_step, N, copy.deepcopy(array_measured_spectra), wavelengths_array, luminances)
task4(lum_to_dv, measurement_step, N, copy.deepcopy(array_measured_spectra), wavelengths_array, luminances)
# print(x(51),y) # to test the interpolation
