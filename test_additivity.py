# RGB
# Yxy eew: [[1.0000e+02 3.3333e-01 3.3333e-01]]
# Yxy primaries: [[1.0000e+02 6.6702e-01 3.1648e-01]
#  [1.0000e+02 2.1180e-01 7.0236e-01]
#  [1.0000e+02 1.5414e-01 6.0887e-02]]
# driver values: [ 76 160  17]
# arguments: [ 76 160  17   0   0   0]
# S. Radiance of mixed color: [[9.7045e-01]]
# Target: [[1.0000e+02 3.3333e-01 3.3333e-01]]
# Result: [[4.0660e+02 3.1075e-01 5.0072e-01]]
# [[4.0660e+02 1.4820e-01 5.3731e-01]]

# RGBAWW
# Yxy eew: [[1.0000e+02 3.3333e-01 3.3333e-01]]
# Yxy primaries: [[1.0000e+02 6.6655e-01 3.1689e-01]
#  [1.0000e+02 2.1203e-01 7.0218e-01]
#  [1.0000e+02 1.5482e-01 6.1488e-02]
#  [1.0000e+02 3.2477e-01 3.3666e-01]
#  [1.0000e+02 5.6569e-01 4.1372e-01]]
# weights: [9.9777e-02 3.6808e-01 5.2368e-02 2.7040e-01 2.0937e-01]
# driver values: [25 93 13 68 53]
# arguments: [25 93 13 68 53  0]
# S. Radiance of mixed color: [[8.0359e-01]]
# Target: [[1.0000e+02 3.3333e-01 3.3333e-01]]
# Result: [[3.1259e+02 3.4088e-01 4.0375e-01]]
# [[3.1259e+02 1.9035e-01 5.0728e-01]]

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

# def changeColors(*args):
#     # R,G,B,A,WW,UV
#     dmx.set_channel(1, args[0])
#     dmx.set_channel(2, args[1])
#     dmx.set_channel(3, args[2])
#     dmx.set_channel(4, args[3])
#     dmx.set_channel(5, args[4])
#     dmx.set_channel(6, args[5])
#     dmx.submit()


# my_port = get_port_by_product_id(24577)
# dmx = Controller(my_port)

# sp.init('jeti')

# # RED
# changeColors(*[255,0,0,0,0,0])
# red_spd = sp.get_spd(manufacturer='jeti')
# red_lum = lx.spd_to_power(red_spd, ptype = 'pu')
# red_XYZ = lx.spd_to_xyz(red_spd, cieobs='1964_10', relative=False)
# print(red_lum)
# print(red_XYZ)

# # BLUE
# changeColors(*[0,255,0,0,0,0])
# blue_spd = sp.get_spd(manufacturer='jeti')
# blue_lum = lx.spd_to_power(blue_spd, ptype = 'pu')
# blue_XYZ = lx.spd_to_xyz(blue_spd, cieobs='1964_10', relative=False)

# print(blue_lum)
# print(blue_XYZ)


# # ADD RED + BLUE
# changeColors(*[255,255,0,0,0,0])
# mix_spd = sp.get_spd(manufacturer='jeti')
# mix_lum = lx.spd_to_power(mix_spd, ptype = 'pu')
# mix_XYZ = lx.spd_to_xyz(mix_spd, cieobs='1964_10', relative=False)

# print(mix_lum)
# print(mix_XYZ)

t_l = 100
Yxy_eew = lx.xyz_to_Yxy(np.array([[t_l,t_l,t_l]]))

