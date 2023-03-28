from DMXEnttecPro import Controller
from luxpy.toolboxes import spectro as sp
import luxpy as lx

def changeColors(R, G, B, W, A, UV):
    dmx.set_channel(1, R)
    dmx.set_channel(2, G)
    dmx.set_channel(3, B)
    dmx.set_channel(4, W)
    dmx.set_channel(5, A)
    dmx.set_channel(6, UV)
    dmx.submit()

# doesn't work. path needs to be replaced
dmx = Controller('/dev/ttyUSB0')  # Typical of Linux

# JETI INITIALIZATION
sp.init('jeti')
spd = sp.get_spd(manufacturer = 'jeti', Tint = 0)

# TASK 2
# Measure spectra of all channels (maybe not UV)
def auto_measure(UV_on):
    array_measured_spectra = [] # in the end it will be [5 or 6]x[255/5]

    if UV_on:
        end = 6 # UV on
    else:
        end = 5 # UV off
        
    for i in range(end):
        channel_array = []
        for j in range(255, 5):
            DMX_arguments_tmp_array = [0]*6
            # we will measure every channel, with 5-10 interval 0-255
            # for every channel, make the arguments for the DMX:
            # eg: for i = 0 (testing RED channel), and j = 155 -> DMX arguments will be: [155, 0, 0, 0, 0, 0]
            # and for i = 3 and j = 255 -> [0, 0, 0, 255, 0, 0]
            DMX_arguments_tmp_array[i] = j
            changeColors(DMX_arguments_tmp_array)
            spd = sp.get_spd(manufacturer = 'jeti', Tint = 0)
            channel_array.append(spd)
        array_measured_spectra.append(channel_array)

    return array_measured_spectra

# call function to measure all channels, with UV off, and print result.
# it should be a 5x51 array. (51, because we took intervals of 5 while testing different driver values for each channel: 0-255)
print(auto_measure(False))


