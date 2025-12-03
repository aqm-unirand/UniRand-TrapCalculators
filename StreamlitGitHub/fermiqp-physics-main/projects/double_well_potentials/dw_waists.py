# %%
#############################################################
#####Invesiigate lattice properties for different waists#####
#############################################################


import sys
import os

from matplotlib import transforms


sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
)

sys.path.append("../..")

from settings import np, plt, pi
from projects.double_well_potentials.doublewells import DW, SL
from lattice.opticallattice import EllipticalGaussianBeam, power2freq

import pprint
import matplotlib as mpl

mpl.rcParams["axes.grid"] = True
mpl.rcParams["lines.marker"] = ""
mpl.rcParams["legend.framealpha"] = 0.75


Erl = 1710
Ers = 6840

# %%
# VERTICAL WAISTS
irwaist_vert = 60e-6
grwaist_vert = 70e-6


irwaist_hor = 120e-6
# grwaists_hor = np.array([110e-6])
grwaists_hor = np.arange(80, 200, 10) * 1e-6

ir_waist = np.sqrt(irwaist_hor * irwaist_vert)
grwaists = np.sqrt(grwaists_hor * grwaist_vert)


###### PARAMS FOR CIRCULAR BEAMS ######

# SL_params = {
#     # "Vs": Vs,
#     # "Vl": Vl,
#     "Vs": 100 * Ers / 4,
#     "Vl": 100 * Ers / 4,
#     "green_waist": grwaists[0],
#     "ir_waist": ir_waist,
#     "phi_s": 0,
#     "phi_l": 0,
# }

###### PARAMS FOR ELLIPTICAL BEAMS ######
SL_params = {
    "Vs_lattice": 100 * Ers,  # None,
    "Vl_lattice": 100 * Ers,  # None,
    "green_waists": [grwaists_hor, grwaist_vert],
    "ir_waists": [irwaist_hor, irwaist_vert],
    "phi_s": 0,
    "phi_l": 0,
}


fig, axs = plt.subplots(nrows=2, ncols=1)
figbarr, axsbarr = plt.subplots(nrows=2, ncols=1)


# Farthest to wathc
edgeindex = 15
gr_powerperbeam = 1.5


ls = "--"
# for x, ls in zip([np.arange(1, 100)]*2.2e-6, ["-", "--", ":"]):
# for x in np.arange(0, 21, 5) * 2.2e-6:
for x in np.array([0, 15, 20]) * 1.2e-6:
    # print(x)
    min_pot = []
    min_pot_c = []
    green_power = []
    green_depth = []

    intra_c = []
    intra_e = []
    inter_c = []
    inter_e = []

    # for grw in grwaists:
    for grw in grwaists_hor:
        # print(grw)
        update_params = SL_params | {
            "green_waists": [grw, grwaist_vert],
            "Vs_lattice": power2freq(
                power=gr_powerperbeam, waist=np.sqrt(grw * grwaist_vert), wavelen=532e-9
            )
            * 4,
        }

        # Define the DW
        center = DW(SL(**update_params, elliptical=True), 0, x=x, z=0)
        edge = DW(SL(**update_params, elliptical=True), edgeindex, x=x, z=0)

        grP = center._DW__SL_instance.green_lattice.beams[0].power
        green_power.append(grP)
        grV = center._DW__SL_instance.green_lattice.beams[0].depth
        green_depth.append(grV * 4)

        min_pot.append(edge.min_pot)
        min_pot_c.append(center.min_pot)

        intra_c.append(center.intrabarrier)
        inter_c.append(center.interbarrier)

        intra_e.append(edge.intrabarrier)
        inter_e.append(edge.interbarrier)

    min_pot = np.array(min_pot)
    min_pot_c = np.array(min_pot_c)

    intra_c = np.array(intra_c)
    intra_e = np.array(intra_e)
    inter_c = np.array(inter_c)
    inter_e = np.array(inter_e)

    axs[0].plot(
        grwaists / 1e-6,
        min_pot_c[:, 0],
        "-",
        label=f"center,  x = {x/1e-6:.2f} um",
    )

    axs[0].plot(
        grwaists / 1e-6,
        min_pot[:, 0],
        linestyle=ls,
        label=f"DW: {edgeindex}, x = {x/1e-6:.2f} um",
        color=axs[0].lines[-1].get_color(),
    )

    axs[1].plot(
        grwaists_hor / 1e-6,
        np.abs(min_pot_c[:, 0] - min_pot_c[:, 1]),
        label=f"center,  x = {x/1e-6:.2f} um",
        color=axs[0].lines[-1].get_color(),
    )

    axs[1].plot(
        grwaists_hor / 1e-6,
        np.abs(min_pot[:, 0] - min_pot[:, 1]),
        linestyle=ls,
        label=f"DW: {edgeindex}, x = {x/1e-6:.2f} um",
        color=axs[0].lines[-1].get_color(),
    )

    axsbarr[0].plot(
        grwaists_hor / 1e-6,
        intra_c[:, 0],
        color=axs[0].lines[-1].get_color(),
        label=f"center,  x = {x/1e-6:.2f} um",
    )

    axsbarr[0].plot(
        grwaists_hor / 1e-6,
        intra_e[:, 0],
        linestyle=ls,
        label=f"DW: {edgeindex}, x = {x/1e-6:.2f} um",
        color=axs[0].lines[-1].get_color(),
    )
    axsbarr[1].plot(
        grwaists_hor / 1e-6,
        inter_c[:, 0],
        color=axs[0].lines[-1].get_color(),
        # label=f"DW,  x = {x/1e-6:.2f} um",
    )

    axsbarr[1].plot(
        grwaists_hor / 1e-6,
        inter_e[:, 0],
        linestyle=ls,
        label=f"DW: {edgeindex}, x = {x/1e-6:.2f} um",
        # color=axs[0].lines[-1].get_color(),
    )


sir = (
    f"IR: $w_{{geom}}^{{IR}}$: {ir_waist/1e-6:.0f} um, $w_{{vert}}$: {irwaist_vert/1e-6:.0f} um,\n"
    + f"$w_{{hor}}^{{IR}}$: {ir_waist**2/irwaist_vert/1e-6:.0f} um\n"
    + f"Power per beam = {center._DW__SL_instance.ir_lattice.beams[0].power:.2f} W\n"
    + f"Lattice depth = {center._DW__SL_instance.ir_lattice.beams[0].depth*4/Ers:.2f} Ers"
)
sgr = f"Green: $w_{{vert}}$: {grwaist_vert/1e-6:.0f} um, "


axs[0].set_xlabel("green geom waist")
axs[0].set_ylabel("$U_L / E_r^s$")
axs[0].set_title(
    # f"x = {center.x/1e-6:.1f} um, z = {center.z/1e-6:.1f} um,"
    f" $\\varphi_{{L}}$ = {center._DW__SL_instance.latt_params['phi_l']/pi:.2f} $\\pi$"
)

# axs[0].text(75, -49, sgr, fontsize=10, transform=ax.transAxes)
axs[0].text(0.2, 0.55, sgr, fontsize=10, transform=axs[0].transAxes)

axs[1].set_xlabel("green horizontal waist")
axs[1].set_ylabel("$|U_L - U_R| / E_r^s$")
axs[1].text(0.20, 0.40, sir, transform=axs[1].transAxes)
axs[1].legend(fontsize=9, ncols=2, loc=4)


axw = axs[1]
axp = axs[1].twiny()
axp.set_xlim(axw.get_xlim())
axp.set_xticks(
    grwaists_hor / 1e-6,
)
axp.set_xticklabels([f"{p/Ers:.2f}" for p in green_depth], rotation=45)
axp.set_xlabel(f"Depth in ERs of the green lattice for P = {gr_powerperbeam:.2f} W")

fig.set_size_inches(5, 6)
fig.tight_layout()


axsbarr[0].set_xlabel("$w_{hor}^{green}$ waist")
axsbarr[0].set_ylabel("Intrawell Barrier / $E_r^s$")
axsbarr[0].set_title(
    # f"x = {center.x/1e-6:.1f} um, z = {center.z/1e-6:.1f} um,"
    f" $\\varphi_{{L}}$ = {center._DW__SL_instance.latt_params['phi_l']/pi:.2f} $\\pi$"
)
axsbarr[0].legend(fontsize=9)

axsbarr[1].text(0.35, 0.55, sir, fontsize=10, transform=axsbarr[1].transAxes)
axsbarr[1].text(0.5, 0.35, sgr, fontsize=10, transform=axsbarr[1].transAxes)

axw = axsbarr[1]
axp = axsbarr[1].twiny()
axp.set_xlim(axw.get_xlim())
axp.set_xticks(
    grwaists_hor / 1e-6,
)
axp.set_xticklabels([f"{p/Ers:.2f}" for p in green_depth], rotation=45)
axp.set_xlabel(f"Depth in ERs of the green lattice for P = {gr_powerperbeam:.2f} W")

axsbarr[1].set_xlabel("$w_{hor}^{green}$ waist")
axsbarr[1].set_ylabel("Interwell barrier / $E_r^s$")
figbarr.set_size_inches(5, 6)
figbarr.tight_layout()


# %%
