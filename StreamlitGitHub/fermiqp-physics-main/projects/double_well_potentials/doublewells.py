# %%

"""
Generates double well potential given a set of four beam.
Investigates minima, maxima in the double well.

Look at dw.ipynb for usage


"""

from __future__ import annotations
import sys
import os

import warnings

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
)

sys.path.append("../..")


from settings import np, plt, pi, hplanck
from lattice.opticallattice import (
    GaussianBeam3D,
    InterferingBeams,
    power2freq,
    EllipticalGaussianBeam,
)
from scipy.signal import argrelextrema


import matplotlib as mpl

mpl.rcParams["axes.grid"] = True
mpl.rcParams["lines.marker"] = ""


def angle_to_vec(theta: float, phi: float = 90) -> np.ndarray:
    """
    returns a unitary vector with the specified angle wrt x-axis
    theta:  in deg,
    phi:    in deg, angle to the lattice direction
    """
    [theta, phi] = np.deg2rad([theta, phi])

    vec = np.array(
        [
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        ]
    )
    return vec / np.linalg.norm(vec)


def createLatticeElliptical(
    green_waists,
    ir_waists,
    phi_s,
    phi_l,
    green_power_per_beam=None,
    ir_power_per_beam=None,
    Vs_lattice=None,
    Vl_lattice=None,
):
    """
    Generates four beams for our y-Superlattice, given their parameters

    Params:
    --------------
    Vs:     LATTICE DEPTH
    Vs:     LATTICE DEPTH
    green_waists = [green waist hor, green waist vert]
    ir_waists = [ir waist hor, ir waist vert]
    phi_s:       phase of the ELECTRIC field of one green beam,
    phi_l:       phase of the ELECTRIC field of one IR beam

    Returns:
    --------------
    green_lattice, ir_lattice:    Instances of InterferingBeams of the green
                                    and IR beams


    TODO: merge with create lattice
    """

    # prop_dir1 = angle_to_vec(0, 14)
    # prop_dir2 = angle_to_vec(0, -14)
    prop_dir1 = angle_to_vec(14, 90)
    prop_dir2 = angle_to_vec(-14, 90)

    axis_dir = angle_to_vec(90 + 14, 90)
    axis_dir2 = angle_to_vec(90 - 14, 90)

    if Vs_lattice is not None:
        gr_depth = Vs_lattice / 4
    else:
        gr_depth = None
    if Vl_lattice is not None:
        ir_depth = Vl_lattice / 4
    else:
        ir_depth = None

    # print(green_power_per_beam)
    gr = EllipticalGaussianBeam(
        waistx=green_waists[0],
        waisty=green_waists[1],
        wavelen=532e-9,
        prop_dir=prop_dir1,
        depth=gr_depth,
        phi=phi_s,
        # wx_axis=[1, 0, 0],
        wx_axis=axis_dir,
        power=green_power_per_beam,
    )

    gr2 = EllipticalGaussianBeam(
        waistx=green_waists[0],
        waisty=green_waists[1],
        wavelen=532e-9,
        prop_dir=prop_dir2,
        depth=gr_depth,
        phi=0,
        wx_axis=axis_dir2,
        power=green_power_per_beam,
    )
    ir = EllipticalGaussianBeam(
        waistx=ir_waists[0],
        waisty=ir_waists[1],
        wavelen=1064e-9,
        prop_dir=prop_dir1,
        depth=ir_depth,
        phi=phi_l,
        wx_axis=axis_dir,
        power=ir_power_per_beam,
    )
    ir2 = EllipticalGaussianBeam(
        waistx=ir_waists[0],
        waisty=ir_waists[1],
        wavelen=1064e-9,
        prop_dir=prop_dir2,
        depth=ir_depth,
        phi=0,  # phi_l,
        wx_axis=axis_dir2,
        power=ir_power_per_beam,
    )
    green_beams = [gr, gr2]
    ir_beams = [ir, ir2]

    green_latt = InterferingBeams(green_beams)
    ir_latt = InterferingBeams(ir_beams)

    return green_latt, ir_latt


def createLattice(Vs, Vl, green_waist, ir_waist, phi_s, phi_l):
    """

    Generates four beams for our y-Superlattice, given their parameters

    Params:
    --------------
    "Vs": Depth of SINGLE green beam (lattice depth is 4*Vs)
    "Vl": Depth of SINGLE IR beam (lattice depth is 4*Vl)
    "green_waist": waist of the green beams,
    "ir_waist":    waist of the IR beams,
    "phi_s":       phase of the ELECTRIC field of one green beam,
    "phi_l":       phase of the ELECTRIC field of one IR beam

    Returns:
    --------------
    green_lattice, ir_lattice:    Instances of InterferingBeams of the green
                                    and IR beams

    """

    # prop_dir1 = angle_to_vec(0, 14)
    # prop_dir2 = angle_to_vec(0, -14)
    prop_dir1 = angle_to_vec(14, 90)
    prop_dir2 = angle_to_vec(-14, 90)

    gr = GaussianBeam3D(
        waist=green_waist,
        wavelen=532e-9,
        prop_dir=prop_dir1,
        depth=Vs,
        phi=phi_s,
    )

    gr2 = GaussianBeam3D(
        waist=green_waist,
        wavelen=532e-9,
        prop_dir=prop_dir2,
        depth=Vs,
        phi=0,
    )
    ir = GaussianBeam3D(
        waist=ir_waist,
        wavelen=1064e-9,
        prop_dir=prop_dir1,
        depth=Vl,
        phi=phi_l,
    )
    ir2 = GaussianBeam3D(
        waist=ir_waist,
        wavelen=1064e-9,
        prop_dir=prop_dir2,
        depth=Vl,
        phi=0,  # phi_l,
    )

    # beams = [gr, gr2, ir, ir2]
    green_beams = [gr, gr2]
    ir_beams = [ir, ir2]

    green_lattice = InterferingBeams(green_beams)
    ir_lattice = InterferingBeams(ir_beams)

    return green_lattice, ir_lattice


class DW:
    dw_size = 2.2e-6
    points_per_dw = 2001
    # points_per_dw = 5001

    def __init__(
        self, SL_instance: SL, index: list, x: float = 0.0, z: float = 0.0
    ) -> None:
        """
        A double well instance


        Params:
        --------------
        SL_instance:    SL
        index:          int
        x:              position along x
        z:              position along z


        If SL_instance x and z coordinates are different than the ones passed
        for the double well, a copy of SL_instance is constructed with the x and
        z coordinates of the double well


        NOTE:
        * `dw.interwell` refers to the value of the maximum potential between the two minima
        * `dw.interbarrier` is the **DIFFERENCE** between the interwell barrier with the minima
        """

        self._index = index
        self.x = x
        self.y = self._gety
        self.z = z
        self.__SL_instance = SL_instance

        if self.x != self.__SL_instance.x or self.z != self.__SL_instance.z:
            warnings.warn(
                "\\nDiscrepancy between the SL coordinates and the DW Coordinates.\n"
                f"Falling back to DW coordinates: x = {self.x}, z = {self.z}"
            )
            self.__SL_instance = SL(
                **SL_instance.latt_params, x=self.x, z=self.z, dwindices=[]
            )

        # For extreme phase shifts, the lattice landscape shifts globally:
        if np.abs(self.__SL_instance.latt_params["phi_s"] - pi) < 1e-17:
            self.shift_y()
        if np.abs(self.__SL_instance.latt_params["phi_l"] - pi / 2) < 1e-17:
            self.shift_y(1 / 2)

        self.lattice_params = self.__SL_instance.latt_params
        self.setProp()

    @property
    def _gety(
        self,
    ):
        """
        calculates the y coordinates of a double well given the index.

        returns:
        --------------
        y:      np.array

        """
        start = self._index * self.dw_size - self.dw_size / 2
        stop = self._index * self.dw_size + self.dw_size / 2
        nsteps = self.points_per_dw
        y = np.linspace(start, stop, nsteps)
        return y

    def shift_y(self, shift=1 / 4):
        """
        shift:      fraction of DW to move
        """
        warnings.warn("\nLattice is shifted")
        self.y -= self.dw_size * shift
        self._pos

        self.setProp()

    @property
    def _pos(self):
        return [self.x, self.y, self.z]

    def getPotential(self):
        return self.__SL_instance.getPotential(self._pos)

    def setMinima(self):
        minima = argrelextrema(
            self._potential,
            np.less,
            # order=1,
            axis=0,
            mode="clip",
        )

        if len(minima[0]) == 2:
            self._min_pot = np.array(self._potential[minima])
            self._min_pos = np.array(self.y[minima])
        elif len(minima[0]) == 1:
            # print(f"{self._index}: Single minima")
            raise ValueError(f"{self._index}: Single minima")
            # self.shift_y()
        return minima

    def getMaxima(self):
        """
        Finds the maxima in the DW potential landscape

        """

        potential = self._potential
        maxima = argrelextrema(potential, np.greater, order=1, axis=0, mode="clip")

        max_pot = potential[maxima]
        max_pos = self.y[maxima]

        # find the interawell:
        cond = (max_pos > self.min_pos[0]) & (max_pos < self.min_pos[1])
        interidx = np.where(cond)
        self.interwell = max_pot[interidx]
        self.interwell_pos = max_pos[interidx]

        # find the intrawell:
        self.intrawell = max_pot[np.where(~cond)]
        self.intrawell_pos = max_pos[np.where(~cond)]

        if self.intrawell.shape[0] == 0:
            self.intrawell = np.array([self._potential[-1]])
            self.intrawell_pos = np.array([self.y[-1]])

        return maxima

    def getBarriers(self):
        """
        Calculates the differences between the minima and maxima of the DW,
        effectively getting the barriers
        """
        self.interbarrier = self.interwell - self.min_pot
        self.intrabarrier = self.intrawell - self.min_pot

        return self.interbarrier, self.intrabarrier
        # self.intrabarrier = self.intrawell - self.min_pot[0]

    def setProp(self):
        self._potential = self.getPotential()
        self.setMinima()
        try:
            # self._min_pos
            # if len(self._min_pos) == 2:
            self.min_pos = np.array(self._min_pos)
            self.min_pot = np.array(self._min_pot)
        except:
            self.min_pos = np.array([None, None])
            self.min_pot = np.array([None, None])

        self.getMaxima()
        try:
            self.imbalance = self.min_pot[0] - self.min_pot[1]
        except:
            self.imbalance = None
        self.getBarriers()

    def plotDW(self, sharexaxis=False, showextrema=True, **kwargs):
        # pot, min_pot, min_pos = self.getPotential()

        pot = self._potential

        if sharexaxis:
            x = np.linspace(-self.dw_size, +self.dw_size, self.points_per_dw) / 1e-6
        else:
            x = self.y / 1e-6
        plt.plot(
            x,
            pot,
            **kwargs,
        )
        if showextrema:
            if (self.min_pos.any() is not None) and (not sharexaxis):
                color = plt.gca().lines[-1].get_color()
                plt.plot(
                    self.min_pos / 1e-6,
                    self.min_pot,
                    "o",
                    color=color,
                )

                plt.vlines(
                    self.interwell_pos / 1e-6,
                    pot.min(),
                    pot.max(),
                    linestyle="--",
                    color=color,
                )

                plt.vlines(
                    self.intrawell_pos / 1e-6,
                    pot.min(),
                    pot.max(),
                    linestyle="-",
                    color=color,
                )

        plt.xlabel("y / um")
        plt.ylabel("U / $Er_s$")
        plt.tight_layout()
        return plt.gcf(), plt.gca()


class SL:
    def __init__(
        self,
        dwindices=[0],
        x=0.0,
        z=0.0,
        elliptical=False,
        **params,
    ) -> None:
        """
        Instance of a Super lattice.
        It collects the superlattice parameters and a list of double wells.
        Every Double well
        (see also the DW class)

        Params:
        --------------
        dwindices:      list of Doublewell indices of interest. Defaults to [0]
        x:              float, x coordinate to look at
        z:              float, z coordinate to look at
        **params        parameters of the lattices. See also the "createLattice" function
                        A dict with the following keys
                            "Vs": Depth of SINGLE green beam (lattice depth is 4*Vs)
                            "Vl": Depth of SINGLE IR beam (lattice depth is 4*Vl)
                            "green_waist": waist of the green beams,
                            "ir_waist":    waist of the IR beams,
                            "phi_s":       phase of the ELECTRIC field of one green beam,
                            "phi_l":       phase of the ELECTRIC field of one IR beam



        """

        self.dwindices = dwindices
        self.x = x
        self.z = z
        self.latt_params = params

        # print(self.latt_params["green_waists"])
        try:
            self.latt_params["green_waists"]
            # if elliptical:
            self.green_lattice, self.ir_lattice = createLatticeElliptical(**params)
        except:
            self.green_lattice, self.ir_lattice = createLattice(**params)
        self.dws = self.getDWs()

    def getPotential(self, pos):
        """Returns the potential in units of Er of the short lattice"""

        # i_ir = self.ir_lattice.getIntensity(pos)
        # i_gr = self.green_lattice.getIntensity(pos)
        p_ir = self.ir_lattice.getPotential(pos)
        p_gr = self.green_lattice.getPotential(pos)

        potential = p_gr + p_ir
        # potential = Vl * i_gr / i_gr.max() - Vl * i_ir / i_ir.max()

        return potential / Ers

    def getDWs(self):
        """returns a dictionary of the double wells"""
        return {f"{i}": DW(self, i, self.x, self.z) for i in self.dwindices}

    def addDW(self, indices):
        """
        adds double well to the Super Lattice. Doesn't allow for duplicates

        Params:
        --------------
        indices:    list of the double wells to add.

        """
        if isinstance(indices, int):
            (
                self.dwindices.append(indices)
                if indices not in self.dwindices
                else self.dwindices
            )
        elif isinstance(indices, list):
            self.dwindices += indices
        else:
            raise TypeError

        self.dwindices.sort()
        self.dwindices = list(dict.fromkeys(self.dwindices))
        self.dws = self.getDWs()
        # self.dws.update([f"i": DW(self, i) for i in indices])

    def getProp(self, attribute):
        """
        puts together the values of "attribute" in one single array

        Params:
        --------------
        attribute:      str, attribute of a double well class

        Returns:
        --------------
        arr:            np.array collecting "attribute" of all DWs in the SL,
                        ordered by DW index
        """
        try:
            self.dws["0"].__dict__[attribute]
        except:
            raise KeyError

        arr = []
        for dw in self.dws.values():
            arr.append(dw.__dict__[attribute])

        arr = np.array(arr)
        return arr


Erl = 1710  # Recoil of the horizontal long lattice
Ers = 6840  # Recooil of the horizontal short lattice

Vl = 1 * Ers / 4
Vs = 1 * Ers / 4

if __name__ == "__main__":
    green_waist = 120e-6
    ir_waist = 120e-6

    phi_long = 0
    phi_short = 0

    SL_params = {
        "Vs": Vs,
        "Vl": Vl,
        "green_waist": green_waist,
        "ir_waist": ir_waist,
        "phi_s": phi_short,
        "phi_l": phi_long,
    }

    lattgr, lattir = createLattice(**SL_params)

    print(lattir.beams[0].power)
    # print(lattir.beams[1].power)
    print(lattgr.beams[0].power)

    SL0 = SL(**SL_params)
    center0 = SL0.dws["0"]
    center0.plotDW()
    # print(lattgr.beams[1].power)
    plt.figure()
    y = np.linspace(-2.2, 2.2, 100) * 1e-6
    plt.plot(-lattgr.getPotential([0, y, 0]))
    plt.plot(-lattir.getPotential([0, y, 0]))
    plt.plot(-lattgr.getPotential([0, y, 0]) - lattir.getPotential([0, y, 0]))

    # %%
    #
    imbalances = []
    min_pot = []
    min_pos = []

    iter_array = np.linspace(0.0, 1 / 2, 5) * pi
    indices = np.arange(
        -50,
        51,
    )
    for phis in iter_array:
        phi_short = phis
        # SL_params = (Vs, Vl, green_waist, ir_waist, phi_long, phi_short)
        SL_params = {
            "Vs": Vs,
            "Vl": Vl,
            "green_waist": green_waist,
            "ir_waist": ir_waist,
            "phi_s": phi_short,
            "phi_l": phi_long,
        }
        SL1 = SL(indices, **SL_params)

        plt.figure("imbalance")
        imbalances.append(SL1.getProp("imbalance"))
        plt.plot(indices, imbalances[-1], marker="", label=phis / pi)
        plt.ylabel("$U_L-U_R$")
        plt.xlabel("DW index")

        min_pot.append(SL1.getProp("min_pot"))
        min_pos.append(SL1.getProp("min_pos"))

    imbalances = np.array(imbalances)
    min_pot = np.array(min_pot)
    min_pos = np.array(min_pos)
    plt.legend()

    # plt.figure("imbs")
    # plt.plot(
    #     iter_array,
    #     imbalances.max(axis=1) - imbalances.min(axis=1),
    #     marker="",
    #     label=f"{phis / pi} $\\pi$",
    # )
    # plt.xlabel("$\\varphi_s$")
    # plt.legend()

    # plt.figure()
    # imbalances = np.array(imbalances)

    # plt.plot(SL1.dwindices, imbalances.T - imbalances[:, imbalances.shape[1] // 2])
    # plt.legend(labels=iter_array / pi)
    # plt.xlabel("DW index")

    # plt.figure("minima")
    fig, axs = plt.subplots(1, 2)

    for i, ph in enumerate(iter_array):
        axs[0].plot(
            min_pos[i, :, 0] / 1e-6,
            min_pot[i, :, 0],
            label=f"$\\varphi_s = {ph/pi} \\pi$",
        )
        axs[1].plot(
            min_pos[i, :, 1] / 1e-6,
            min_pot[i, :, 1],
            label=f"$\\varphi_s = {ph/pi} \\pi$",
        )

    axs[1].set_title("Right")
    axs[0].set_title("Left")

    for ax in axs:
        # ax.set_ylim(-1, 0)
        ax.set_xlabel("y / um")
    axs[0].legend()

    fig.suptitle("Minima")
    fig.tight_layout()

    # %%
    center = DW(SL1, 0)
    edgeright = DW(SL1, 20)
    edgeleft = DW(SL1, -20)

    edgeright.plotDW(sharexaxis=False)
    plt.gca().spines["top"].set_color(plt.gca().lines[-1].get_color())
    plt.gca().tick_params(axis="x", colors=plt.gca().lines[-1].get_color())
    plt.twiny()
    edgeleft.plotDW(sharexaxis=False, color="C2")
    plt.gca().tick_params(axis="x", colors=plt.gca().lines[-1].get_color())
    plt.gca().spines["bottom"].set_position(("axes", 1.1))
    plt.twiny()
    center.plotDW(color="C1")
    plt.gcf().tight_layout()
    plt.gca().tick_params(axis="x", colors=plt.gca().lines[-1].get_color())

    # %%

    # %%

    lol = SL1.getProp("min_pot")
    lel = SL1.getProp("min_pos")

    # %%
    # plt.plot(SL1.dwindices, lol, marker=".")
    plt.figure()
    plt.plot(lel, lol, marker=".")
    # %%
    print(edgeright.min_pot)
    print(edgeright.min_pos)

    # %%

    plt.figure()
    edgeright.plotDW(sharexaxis=False)
    plt.gca().spines["top"].set_color(plt.gca().lines[-1].get_color())
    plt.gca().tick_params(axis="x", colors=plt.gca().lines[-1].get_color())
    plt.twiny()
    edgeleft.plotDW(sharexaxis=False, color="C2")
    plt.gca().tick_params(axis="x", colors=plt.gca().lines[-1].get_color())
    plt.gca().spines["bottom"].set_position(("axes", 1.1))
    plt.twiny()
    center.plotDW(color="C1")
    plt.gcf().tight_layout()
    plt.gca().tick_params(axis="x", colors=plt.gca().lines[-1].get_color())

    # plt.figure()
    # p = np.concatenate((DW(SL1, -50).getPotential()[0], DW(SL1, -49).getPotential()[0]))
    # x = np.concatenate((DW(SL1, -50)._gety, DW(SL1, -49)._gety))
    # pt = SL1.getPotential([0, x, 0])
    # plt.plot(x, pt)
    # plt.plot(x, p)

    # %%
    edgeright.getMaxima()
    plt.show()

# %%
