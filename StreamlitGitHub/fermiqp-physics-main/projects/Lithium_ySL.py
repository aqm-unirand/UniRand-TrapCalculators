# %%
import sys
import os

# print("CWD\n", os.getcwd())
# print(os.listdir())


# print("\nSYS PATH", sys.path)
# sys.path.insert(0, "/..")
# sys.path.insert(0, os.getcwd())
# sys.path.append(0, "/..")

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    ),
)

from settings import *
from lattice.opticallattice import *


# %%
"""
1064: 
powers before fibres: 9W
we can assume that they lose some power to fibre coupling and polarization cleaning.

532:

"""


lithiumIr = OpticalLatticeAngle(
    waist=350e-6,
    half_angle=13.37,
    # power=40 / 4/3,
    power=9 * 0.7 / 2,
)

lithiumGr = OpticalLatticeAngle(
    waist=120e-6,
    half_angle=13.37,
    power=2,
    wavelen=532e-9,
)


# %%
print("GREEN\n")
lithiumGr.print_params()

print("\n=========================\n")
print("IR\n")
lithiumIr.print_params()
# %%
