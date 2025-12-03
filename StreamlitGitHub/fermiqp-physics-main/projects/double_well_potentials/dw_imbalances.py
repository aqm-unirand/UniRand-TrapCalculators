##############################################
#####calculate imbalances for given waists#####
##############################################


# %%

import sys
import os


sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
)

sys.path.append("../..")

from settings import np, plt, pi, hplanck
from projects.double_well_potentials.doublewells import DW, SL
from lattice.opticallattice import EllipticalGaussianBeam, power2freq

import pprint
import matplotlib as mpl

mpl.rcParams["axes.grid"] = True
mpl.rcParams["lines.marker"] = ""
mpl.rcParams["legend.framealpha"] = 0.75
mpl.rcParams["legend.fontsize"] = 10


Erl = 1710
Ers = 6840

# VERTICAL WAISTS
irwaist_vert = 65e-6
grwaist_vert = 65e-6


irwaist_hor = 110e-6
# grwaists_hor = np.arange(80, 200, 10)*1e-6
grwaists_hor = 110e-6

ir_waist = np.sqrt(irwaist_hor * irwaist_vert)
grwaists = np.sqrt(grwaists_hor * grwaist_vert)


imbalances = []
min_pot = []
min_pos = []
inter = []
intra = []


# 100 * Ers / 4,
# 100 * Ers / 4,


###### PARAMS FOR CIRCULAR BEAMS ######
# Vl = (power2freq(power=1.5, waist=ir_waist, wavelen=1064e-9),)
# Vs = (power2freq(power=1.5, waist=ir_waist, wavelen=532e-9),)
# SL_params = {
#     # "Vs": Vs,
#     # "Vl": Vl,
#     "Vs": Vs,
#     "Vl": Vl,
#     "green_waist": grwaists,
#     "ir_waist": ir_waist,
#     "phi_s": 0,
#     "phi_l": 0,
# }


##### PARAMS FOR ELLIPTICAL BEAMS ######
SL_params = {
    # "Vs_lattice": 100 * Ers,  # None,
    # "Vl_lattice": 100 * Ers,  # None,
    "green_waists": [grwaists_hor, grwaist_vert],
    "ir_waists": [irwaist_hor, irwaist_vert],
    "phi_s": 0,
    "phi_l": 0,
    "green_power_per_beam": 1.5,
    "ir_power_per_beam": 1.5,
}

pprint.pprint(SL_params)

iter_array = np.linspace(0, 1 / 2, 5, endpoint=True) * pi
indices = np.arange(-25, 26)
for phil in iter_array:
    update_params = SL_params | {"phi_l": phil}

    SL1 = SL(indices, **update_params)
    # plt.figure("DWs")
    # SL1.dws["0"].plotDW()
    # SL1.dws["1"].plotDW(**{"color": plt.gca().lines[-1].get_color()})

    plt.figure("imbalance")
    imbalances.append(SL1.getProp("imbalance"))
    plt.plot(
        indices,
        imbalances[-1],
        marker="",
        label=f"$\\varphi_{{IR}}  = {phil / pi:.3} \\pi$",
    )
    plt.ylabel("$U_L-U_R$ / $E_R^s$")
    plt.xlabel("DW index")

    min_pot.append(SL1.getProp("min_pot"))
    min_pos.append(SL1.getProp("min_pos"))
    inter.append(SL1.getProp("interbarrier"))
    intra.append(SL1.getProp("intrabarrier"))


plt.legend()


imbalances = np.array(imbalances)
min_pot = np.array(min_pot)
min_pos = np.array(min_pos)
inter = np.array(inter)
intra = np.array(intra)

plt.title("Imbalance across the lattice")
plt.tight_layout()

# %%
fig, axs = plt.subplots(nrows=1, ncols=2)

for i, ph in enumerate(iter_array):
    axs[0].plot(
        min_pos[i, :, 0] / 1e-6,
        min_pot[i, :, 0],
        label=f"$\\varphi_{{IR}} = {ph/pi:.3f} \\pi$",
    )
    axs[1].plot(
        min_pos[i, :, 1] / 1e-6,
        min_pot[i, :, 1],
        label=f"$\\varphi_{{IR}} = {ph/pi:.3f} \\pi$",
    )

axs[1].set_title("Right")
axs[0].set_title("Left")

for ax in axs:
    ax.set_ylim(min_pot.min(), 0)
    ax.set_xlabel("y / um")
    ax.set_ylabel("U / $E_r^s$")
axs[0].legend(framealpha=0.8)

fig.suptitle("Minima")
fig.tight_layout()
# %%

plt.figure("Difference")
plt.subplot(121)
plt.plot(
    indices,
    imbalances[:, :].T,
    marker="",
    # indices, imbalances[:,:].T, marker="",
)
plt.ylabel("$U_L-U_R$ / $E_R^S$")
plt.xlabel("DW index")
# plt.legend(labels=[f"$\\varphi_{{IR}}  = {i/pi:.3f} \\pi$" for i in iter_array[:]])
plt.title("Imbalances")

plt.subplot(122)
plt.plot(
    indices,
    imbalances[:, :].T - imbalances[:, imbalances.shape[1] // 2],
    marker="",
    # indices, imbalances[:,:].T, marker="",
)
plt.ylabel("$\\Delta U -\\Delta U_0$  / $E_R^S$")
plt.xlabel("DW index")
plt.legend(
    labels=[f"$\\varphi_{{IR}}  = {i/pi:.3f} \\pi$" for i in iter_array[:]],
    loc=(1.1, 0.25),
)

plt.title("Difference from the central well")

plt.gcf().set_size_inches(7.5, 3)
# plt.xlim(-20, 20)
plt.tight_layout()

# %%

plt.figure("abs")
plt.subplot(121)
plt.plot(
    indices,
    imbalances[:, :].T,
    marker="",
    # indices, imbalances[:,:].T, marker="",
)
plt.ylabel("$U_L-U_R$ / $E_R^S$")
plt.xlabel("DW index")
# plt.legend(labels=[f"$\\varphi_{{IR}}  = {i/pi:.3f} \\pi$" for i in iter_array[:]])
plt.legend(
    labels=[f"$\\varphi_{{IR}}  = {i/pi:.3f} \\pi$" for i in iter_array[:]],
    loc=(1.1, 0.25),
)
plt.gcf().set_size_inches(5, 3)
plt.title("Imbalances")

# %%
plt.figure("Normalized")
plt.subplot(121)
plt.plot(
    indices,
    imbalances[:, :].T / np.abs(imbalances).max(axis=1),
    marker="",
    # indices, imbalances[:,:].T, marker="",
)
plt.ylabel("$U_L-U_R$")
plt.xlabel("DW index")
plt.legend(labels=[f"$\\varphi_{{IR}}  = {i/pi:.3} \\pi$" for i in iter_array[:]])
plt.axhline(-0.75, linestyle="--", color="steelblue")
plt.axvline(15, linestyle="--", color="steelblue")
plt.title("Normalized to the maximum")


plt.subplot(122)
plt.plot(
    indices,
    imbalances[:, :].T - imbalances[:, imbalances.shape[1] // 2],
    marker="",
    # indices, imbalances[:,:].T, marker="",
)
plt.ylabel("$\\Delta U -\\Delta U_0$  / $E_R^S$")
plt.xlabel("DW index")
plt.legend(labels=[f"$\\varphi_{{IR}}  = {i/pi:.3f} \\pi$" for i in iter_array[:]])
# plt.axhline(-0.75, linestyle="--", color="steelblue")
plt.axvline(15, linestyle="--", color="steelblue")
plt.axvline(-15, linestyle="--", color="steelblue")
plt.title("Difference from the central well")

plt.tight_layout()

# %%
plt.figure("Normsep")


fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)

axs[0, 0].set_title("Balanced, normalized", fontsize=10)
axs[0, 0].plot(
    indices,
    imbalances[:1, :].T / np.abs(imbalances[:1]).max(axis=1),
    marker="",
    # indices, imbalances[:,:].T, marker="",
)
axs[0, 0].plot(
    indices,
    imbalances[-1:, :].T / np.abs(imbalances[-1:]).max(axis=1),
    marker="",
    # indices, imbalances[:,:].T, marker="",
)
axs[0, 0].legend(
    labels=[f"$\\varphi_{{IR}}  = {i/pi:.3} \\pi$" for i in iter_array[[0, -1]]]
)


axs[1, 0].set_title("Imbalanced, normalized", fontsize=10)
axs[1, 0].plot(
    indices,
    imbalances[1:-1, :].T / np.abs(imbalances[1:-1]).max(axis=1),
    marker="",
    # indices, imbalances[:,:].T, marker="",
)
axs[1, 0].legend(
    labels=[f"$\\varphi_{{IR}}  = {i/pi:.3} \\pi$" for i in iter_array[1:-1]]
)


axs[0, 0].set_ylabel("$\\Delta U$ / $\\Delta U_{max}$")
axs[1, 0].set_ylabel("$\\Delta U$ / $\\Delta U_{max}$")
axs[1, 0].set_xlabel("DW index")
# plt.legend(labels=[f"$\\varphi_{{IR}}  = {i/pi:.3} \\pi$" for i in iter_array[:]])
# plt.axhline(-0.75, linestyle="--", color="steelblue")
# plt.axvline(15, linestyle="--", color="steelblue")
# plt.title("Normalized to the maximum")


axs[0, 1].plot(
    indices,
    imbalances[[0, -1], :].T - imbalances[[0, -1], imbalances.shape[1] // 2],
    marker="",
)
axs[0, 1].plot(
    indices,
    imbalances[-1, :].T - imbalances[-1, imbalances.shape[1] // 2],
    marker="",
    # indices, imbalances[:,:].T, marker="",
)
axs[0, 1].legend(
    labels=[f"$\\varphi_{{IR}}  = {i/pi:.3} \\pi$" for i in iter_array[[0, -1]]]
)
axs[1, 1].plot(
    indices,
    (imbalances[1:-1, :].T - imbalances[1:-1, imbalances.shape[1] // 2]),
    marker="",
    # indices, imbalances[:,:].T, marker="",
)
axs[1, 1].legend(
    labels=[f"$\\varphi_{{IR}}  = {i/pi:.3} \\pi$" for i in iter_array[1:-1]]
)
axs[1, 1].set_ylabel("$\\Delta U -\\Delta U_0$  / $E_R^S$")
axs[0, 1].set_ylabel("$\\Delta U -\\Delta U_0$  / $E_R^S$")
axs[1, 1].set_xlabel("DW index")
# axs[0, 1].legend(

#     labels=[f"$\\varphi_{{IR}}  = {i/pi:.3f} \\pi$" for i in iter_array[:]]
# )
# plt.axhline(-0.75, linestyle="--", color="steelblue")
axs[1, 0].axhline(-0.75, linestyle="--", color="steelblue")
# axs[1, 0].axvline(20, linestyle="--", color="steelblue")
# axs[1, 0].axvline(-20, linestyle="--", color="steelblue")
# axs[1, 1].axvline(20, linestyle="--", color="steelblue")
# axs[1, 1].axvline(-20, linestyle="--", color="steelblue")

axs[0, 1].set_title("Difference from the central well", fontsize=10)
axs[1, 1].set_title("Difference from the central well", fontsize=10)

plt.tight_layout()

# %%
plt.figure("Barriers")
plt.subplot(121)
# plt.plot(

#     indices,
#     inter[:, :, 0].T,
#     marker="",
#     label="Interwell barrier",
#     # indices, imbalances[:,:].T, marker="",
# )
plt.plot(
    indices,
    inter[:, :, 1].T,
    marker="",
    label="Interwell barrier",
    # linestyle="--",
    # indices, imbalances[:,:].T, marker="",
)
# plt.legend(labels=[f"$\\varphi_{{IR}}  = {i/pi:.3f} \\pi$" for i in iter_array[:]])
plt.title("Interwell Barriers")
plt.ylabel("$U_{inter}-U_R$ \\ Ers")
plt.ylim(0, 200)

plt.subplot(122)
# plt.plot(
#     indices,
#     intra[:, :, 0].T,
#     linestyle="--",
#     # indices, imbalances[:,:].T, marker="",
# )
plt.gca().set_prop_cycle(None)

plt.plot(
    indices,
    intra[:, :, 1].T,
    # linestyle="--",
    # color=plt.gca().lines[-1].get_color(),
)
plt.legend(
    labels=[f"$\\varphi_{{IR}}  = {i/pi:.3f} \\pi$" for i in iter_array[:]],
    loc=(1.1, 0.25),
)

plt.ylabel("$U_{intra}-U_R$ \\ Ers")
plt.ylim(0, 200)
plt.xlabel("DW index")
# plt.legend()
plt.title("Intrawell Barrier")
plt.gcf().set_size_inches(7.5, 3)
plt.tight_layout()


plt.show()
# %%
SL1.dws["0"].plotDW(**{"label": "center"})
plt.gca().tick_params(axis="x", colors=plt.gca().lines[-1].get_color())
plt.gca().spines["bottom"].set_color(plt.gca().lines[-1].get_color())
plt.tight_layout()
plt.twiny()
SL1.dws["15"].plotDW(**{"color": "C1", "label": "15"})
plt.gca().spines["top"].set_color(plt.gca().lines[-1].get_color())
plt.gca().tick_params(axis="x", colors=plt.gca().lines[-1].get_color())

plt.gcf().legend(loc=5)
plt.tight_layout()

plt.gcf().set_size_inches(4, 3)

plt.plot(SL1.dwindices, SL1.getProp("interbarrier"))
SL1.getProp("interwell")[5] - SL1.getProp("min_pot")[5]

# %%
