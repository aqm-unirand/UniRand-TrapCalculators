import warnings
from math import pi

import numpy as np
import pandas as pd

### Definition of natural constants ###
hplanck = 6.63e-34
hbar = hplanck / 2 / pi
mass = 6 * 1.67e-27
kB = 1.38e-23
epsilon0 = 8.8541878128e-12
c = 299792458
Gamma = 2 * pi * 5.8724e6
omega0 = 2 * pi * 446.799677e12


### Conversion functions ###
def temp2freq(T):
    """
    Converts temperature values from Kelvin to frequency in Hz
    """
    return kB * T / hplanck


def freq2temp(freq):
    """
    Converts frequency values from Herz into Kelvin
    """
    return hplanck * freq / kB


def power2freq(power=1.0, waist=60e-6, waisty=None, wavelen=1.064e-6):
    """Calculates the trap depth U in units of Hz for a TLS created by a single beam.
    For a lattice this number needs to be multiplied by an eta_boost parameter that
    takes into account the intenstiy boost from the interference
    Example: 1) For two beams interfering eta_boost = (1+1)**2 = 4
             2) For our pinning lattice the power ratio goes down by a factor of t=0.91
                for every pass through the glass cell. Hence the correct boost factor is

                eta_boost = (t + t**3 + t**5 + t**7)**2

                Note that in this definition we include losses on the first entry on the glass cell.
                Therefore we can plug in the beam power that we measure in the experiment just before
                it enters the glass cell.

    Parameters
    ========
    power       Power of the beam in W
    waist       Guassian beam waist in m
    waisty      For elliptical beams (optional), secondary Guassian beam waist in m
    wavelen     wavelength in m

    Returns
    ========
    U:           Trap depth in units of Hz (i.e. trap depth U in Joule divided by Planck constant h)

    TODOs
    ========
    implement calculation for alkali (see qcalculate from Li1.0)
    """
    if waisty is None:
        waisty = waist
    Imax = 2 * power / (pi * waist * waisty)
    omega = 2 * pi * c / wavelen

    Gamma_vals = [2 * pi * 5.8724e6, 2*pi*.754e6]
    omega0_vals = [2 * pi * 446.799677e12, 2*pi*9.287925696594426e14]

    U = 0
    for omega0, Gamma in zip(omega0_vals, Gamma_vals):

        U += (
            3
            * pi
            * c**2
            / 2
            / omega0**3
            * Gamma
            * (1 / (omega0 - omega) + 1 / (omega0 + omega))
            * Imax
        )
    return U / hplanck


### Class definitions fo Dipole traps and lattices ###
class DipoleTrap:
    """Optical dipole trap created by a single beam

    Parameters
    =========
    waist       Gaussian beam waist in m
    waisty      For elliptical beams (optional), secondary Guassian beam waist in m
    wavelen     Wavelength in m
    U           Trap depth in Hz

    """

    def __init__(
        self,
        waist=60e-6,
        waisty=None,
        wavelen=1.064e-6,
        U=1,
    ) -> None:
        self.waist = waist
        self.wavelen = wavelen
        self.zR = pi * self.waist**2 / self.wavelen
        self.U = U

        if waisty:
            self.waisty = waisty
            self.zRy = pi * self.waisty**2 / self.wavelen

    def trap_freq(self, U=None, unit="Hz", axis="radial"):
        """Reuturns trap frequencies for a dipole trap instance

        Parameters
        =========
        U           Trap depth in Hz
        unit        optional, accepted values are "Hz" and "K"
                    Default is "Hz"
        axis        optional, accepted values are "axial" and "radial"

        Returns
        ========
        f           trap frequency in Hz.
                    For elliptical beams, returns a list of two trap frequencies if axis
                    is "radial"
        """
        if U is None:
            U = self.U

        if unit == "Hz":
            pass
        elif unit == "K":
            U = temp2freq(U)

        if axis == "radial":
            try:
                self.waisty
            except Exception:
                frequency = (
                    (-1)**(U<0)*(4 * abs(U) * hplanck / mass / self.waist**2) ** 0.5 / 2 / pi
                )
            else:
                frequency = [
                    (-1)**(U<0)*(4 * abs(U) * hplanck / mass / self.waist**2) ** 0.5 / 2 / pi,
                    (-1)**(U<0)*(4 * abs(U) * hplanck / mass / self.waisty**2) ** 0.5 / 2 / pi,
                ]
        elif axis == "axial":
            try:
                self.waisty
            except Exception:
                frequency = (-1)**(U<0)*(
                    (2 * abs(U) * hplanck / mass / self.zR**2) ** 0.5 / 2 / pi
                )
            else:
                # Use expression derived in Lucas mathematice notebook
                wz0 = (2 * pi * self.waist**2 * self.waisty**2) / (
                    self.wavelen * np.sqrt(self.waist**4 + self.waisty**4)
                )
                frequency = (-1)**(U<0)*(4 * abs(U) * hplanck / mass / wz0**2) ** 0.5 / 2 / pi
        return frequency

    def a_ho(self, U=None):
        """Returns the size of the ground state of the harmonic oscillator in
        the axial and radial directions.

        Returns
        =====
        a_ho        List with 2 entries for circular trap, and 3 entries for
                    an elliptical trap. unit (m)
        """
        if U is None:
            U = self.U
        try:
            self.waisty
        except Exception:
            a_ho = [
                np.sqrt(
                    hbar / mass / self.trap_freq(U=U, axis="radial") / 2 / pi
                ),
                np.sqrt(
                    hbar / mass / self.trap_freq(U=U, axis="axial") / 2 / pi
                ),
            ]
        else:
            a_ho = [
                np.sqrt(
                    hbar
                    / mass
                    / self.trap_freq(U=U, axis="radial")[0]
                    / 2
                    / pi
                ),
                np.sqrt(
                    hbar
                    / mass
                    / self.trap_freq(U=U, axis="radial")[1]
                    / 2
                    / pi
                ),
                np.sqrt(
                    hbar / mass / self.trap_freq(U=U, axis="axial") / 2 / pi
                ),
            ]

        return a_ho
    
    def fermi_temp(self,N):
        '''Calculate Fermi temperature for a given particle number'''
        return freq2temp((6*N/2*self.trap_freq(axis="radial")[0]*
                self.trap_freq(axis="radial")[1]*
                self.trap_freq(axis="axial"))**(1/3))
    
    def print_params(self,N):    

        omegax = self.trap_freq()[0]
        omegay = self.trap_freq()[1]
        omegaz = self.trap_freq(axis="axial")


        data = {
            "parameters": [
                r"Trap frequency (horizontal): ",
                r"Trap frequency (vertical): ",
                r"Trap frequency (axial): ",
                r"Trap depth: ",
                # r"$T_f$: ",
            ],
            "sheet trap (Hz)": [
                f"{self.trap_freq(axis='radial')[0]:.1f} Hz",
                f"{self.trap_freq(axis='radial')[1]:.1f} Hz",
                f"{self.trap_freq(axis='axial'):.1f} Hz",
                f"{freq2temp(self.U)*1e3:.3f} mK",
                # f"{self.fermi_temp(N)*1e6:.3f} uK",
            ],
        }
        df = pd.DataFrame(data)

        return df


class OpticalLattice1D:
    """Optical one-dimensional lattice with known spacing and trap depth

    Parameters
    =========
    a       lattice spacing in m
    U       trap depth in units of

    Properties
    =========
    Er      recoil energy in units of Hz

    """

    def __init__(self, a=1.1e-6, U=1) -> None:
        self.a = a
        self.Er = hplanck / 8 / mass / (self.a) ** 2
        self.U = U

    def t(self, U=None):
        """Returns tunneling rate in units of Hz
        note: only valid for deep lattices, error smaller 10% if U > 15Er
        """
        if U is None:
            U_in_Er = self.U / self.Er
        else:
            U_in_Er = U / self.Er
        return (
            4
            / np.pi**0.5
            * (U_in_Er) ** (3 / 4)
            * np.exp(-2 * (U_in_Er) ** 0.5)
            * self.Er
        )

    def trap_freq(self, U=None, unit="Hz"):
        if U is None:
            U = self.U

        if unit == "Hz":
            pass
        elif unit == "K":
            U = temp2freq(U)

        return 1 / 2 / self.a * np.sqrt(2 * np.abs(U) * hplanck / mass)

    def exactharmonics(self, n=0, U=None):
        if U is None:
            U = self.U

        pass

    def a_ho(self, U=None):
        if U is None:
            U = self.U

        return np.sqrt(hbar / mass / self.trap_freq(U=U) / 2 / pi)


class OpticalLatticeAngle:
    """Optical lattice created by two beams interfering under an angle

    Parameters
    =========
    waist       Gaussian beam waist in m
    waisty      For elliptical beams (optional), secondary Guassian beam waist in m
    half_angle  Half the interference angle between the
                two beams creating the lattice in degrees
    power       The power in each individual beam in W
    wavelen     Wavelength in m
    U           depth in Hz

    """

    def __init__(
        self,
        waist=60e-6,
        waisty=None,
        half_angle=10,
        power=1,
        wavelen=1.064e-6,
        # U=1,
    ):
        if waisty is not None:
            self.waisty = waisty
        else:
            self.waisty = waist

        self.waist = waist
        self.half_angle = half_angle
        self.power = power
        self.wavelen = wavelen
        self.a = self.wavelen / (2 * np.sin(np.pi / 180 * self.half_angle))
        self.Er = hplanck / 8 / mass / (self.a) ** 2  # Hz
        # self.U = U
        self.U = (
            power2freq(
                self.power,
                self.waist,
                self.waisty,
                self.wavelen,
            )
            * 4
        )  # Hz

        self.U_abs = np.abs(
            self.U
        )  # useful for blue detuned lattices, where the trapping is negative

        self.U_in_Er = self.U / self.Er

    def t(self, U=None):
        """Returns tunneling rate in units of Hz

        Parameters
        ----
        U   optional, in Hz

        note: only valid for deep lattices, error smaller 10% if U > 15Er
        """
        if U is None:
            U_in_Er = np.abs(self.U_in_Er)
        else:
            U_in_Er = np.abs(U / self.Er)
        return (
            4
            / np.pi**0.5
            * (U_in_Er) ** (3 / 4)
            * np.exp(-2 * (U_in_Er) ** 0.5)
            * self.Er
        )

    def on_site_freq(self, U=None, unit="Hz"):
        """Returns on site frequency
        Parameters:


        """

        if U is None:
            U = self.U_abs

        if unit == "Hz":
            pass
        elif unit == "K":
            U = temp2freq(U)

        return 2 *np.sqrt(np.abs(U) / self.Er) * self.Er

    def a_ho(self, U=None):
        if U is None:
            U = self.U
        f = np.abs(self.on_site_freq(U=U))
        return np.sqrt(hbar / mass / f / 2 / pi)

    def __str__(self):
        return f"Lattice with interference half-angle {self.half_angle}"

    def print_params(self):
        """Does the same as vars() but prettier"""

        print(f"Power per beam: {self.power:.1f} W")
        print(f"Waists: {self.waist /1e-6:.3f}, {self.waisty/1e-6:.3f} um")
        print(f"Lattice spacing: {self.a/1e-6:.3f} um")
        print(f"Recoil energy: {self.Er:.3f} Hz")
        print(
            f"Depth: {self.U /1e3:.3f} kHz, {self.U_in_Er:.3f} Er, {freq2temp(self.U)/1e-6:.3f} uK"
        )
        print(
            f"On site: {self.on_site_freq() /1e3:.3f} kHz, {freq2temp(self.on_site_freq())/1e-6:.3f} uK"
        )


class GaussianBeam3D:
    """Gaussian Beams in 3D space

    Parameters:
        waist:       Gaussian beam waist in m
        focuspos:    Positon of the focus
        wavelen:     Wavelength in m
        prop_dir:    Propagation direction
        polar_dir:   Polarization direction (linear)
        power:       Optional, The power in each individual beam in W
        depth:       Optional, Depth of the dipole trap crated by the beam in Hz.
                        NOTE: There are no checks on physical meanings of values





    TODO:
        Implement elliptical beams
        Scale E field with power
        Check all phase and wavefront curvature
    """

    def __init__(
        self,
        waist: np.ndarray,
        focuspos=np.array([0, 0, 0]),
        wavelen=1.064e-6,
        prop_dir=np.array([1, 0, 0]),
        polar_dir=np.array([0, 0, 1]),
        phi=0.0,
        power=None,  # W
        depth=None,
    ) -> None:
        """
        Parameters:
            waist:       Gaussian beam waist in m
            focuspos:    Positon of the focusm, Defaults to [0,0,0]
            wavelen:     Wavelength in m. Defaults to 1064e-9
            prop_dir:    Propagation direction
            polar_dir:   Polarization direction (linear)
            power:       Optional, The power in each individual beam in W
            depth:       Optional, Depth of the dipole trap crated by the beam in Hz.
                            NOTE: There are no checks on physical meanings of values

        """
        self.wavelen = wavelen
        self.waist = waist
        self.k = 2 * pi / self.wavelen

        self.focuspos = focuspos
        self.x0, self.y0, self.z0 = (
            focuspos[0],
            focuspos[1],
            focuspos[2],
        )
        self.zR = pi * self.waist**2 / self.wavelen

        self.prop_dir = prop_dir / np.linalg.norm(prop_dir)
        self.polar_dir = polar_dir / np.linalg.norm(polar_dir)

        self.phi = phi
        self.power = power
        self.depth = depth

        if power is not None and depth is not None:
            raise ValueError("Give either Power or depth, not both")
        if depth is None and power is not None:
            self.depth = power2freq(power, waist=self.waist)
        if depth is not None and power is None:
            # depth is linear wih power
            self.power = depth / power2freq(power=1, waist=self.waist)

        # theta = np.arctan(prop_dir[0]
        # self.rotmat = np.array([[np.cos(theta), np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    def field(self, pos):
        """
        Calculates the Electric field in a.u.

        Paramters:
        ----------
        pos:    array or list [X,Y] or [X,Y,Z]
            coordinates in 3D space


        """

        warnings.warn("E field calculation doesn't account for power")

        if len(pos) == 2:
            X, Y = pos
            Z = 0
        else:
            X, Y, Z = pos

        # distance from the focus position:
        R = (
            (X - self.x0) ** 2 + (Y - self.y0) ** 2 + (Z - self.z0) ** 2
        ) ** 0.5

        # distance along the axial direction:
        z_axial = (
            self.prop_dir[0] * (X - self.x0)
            + self.prop_dir[1] * (Y - self.y0)
            + self.prop_dir[2] * (Z - self.z0)
        )

        # Distance along the radial direction
        r = np.sqrt(R**2 - z_axial**2)

        waist = self.waist * np.sqrt(1 + z_axial**2 / self.zR**2)

        # wavefront_curv = z_axial / (1 + (self.zR / z_axial) ** 2)
        # wavefront_curv = z_axial / (
        #     1
        #     + np.divide(  # this takes care of division by 0
        #         (self.zR / z_axial),
        #         where=z_axial != 0,
        #         out=np.zeros_like(z_axial),
        #         out=np.
        #     )
        #     ** 2
        # )

        # factor = np.ewavefront_curv

        gouy_phase = np.arctan(z_axial / self.zR)
        # print(wavefront_curv)

        # return field
        return (
            self.waist
            / waist
            * np.exp(-(r**2) / waist**2)
            * np.exp(
                -1j
                * (
                    X * self.prop_dir[0]
                    + Y * self.prop_dir[1]
                    + Z * self.prop_dir[2]
                )
                * self.k
                + 1j * gouy_phase
                - 1j * self.phi
            )
            # * np.exp(+-1j * self.k * r**2 / wavefront_curv / 2)
        )

    def getIntensity(self, pos):
        """Calculates the intensity of the beam

        Parameters:
            pos:    array of the form [X,Y,Z]

        """
        Imax = 2 * self.power / pi / self.waist**2
        # Imax = intensity(
        #     self.power,
        #     radial_pos=0,
        #     radius=self.waist,
        # )
        # TODO: why the hell is this so over complicates
        return np.abs(self.field(pos) ** 2) * Imax
        # intensity = np.zeros_like(pos)


class InterferingBeams:
    """Constructs interference patterns from gaussian beams

    Parameters:
        beams:   list or array of beam

    """

    def __init__(self, beams):
        # if type(beams) != list and not (isinstance(beams, GaussianBeam3D)):
        if not (isinstance(beams, list)) and not (
            isinstance(beams, GaussianBeam3D)
        ):
            raise Warning(
                "Beams need to be a GaussianBeam3D or a list of them"
            )
        elif not (isinstance(beams, GaussianBeam3D)):
            for el in beams:
                if not (isinstance(el, GaussianBeam3D)):
                    raise TypeError(
                        "Each entry must be an instance of GaussianBeam3D"
                    )
        self.beams = beams

    def getIntensity(self, pos):
        field = np.zeros_like(self.beams[0].field(pos))
        for beam in self.beams:
            f = beam.field(pos)
            # scaling_factor = (
            #     2
            #     * beam.power
            #     / pi
            #     / beam.waist**2
            #     / np.max(np.abs(field**2))
            # )

            # Imax = beam.getIntensity(pos).max()
            Imax = 2 * beam.power / pi / beam.waist**2
            field += f * np.sqrt(Imax)

        intensity = np.abs(field) ** 2

        return intensity


def intensity(
    power: np.ndarray,
    radius: float,
    radial_pos: float = 0,
):
    """Returns the intensity of a gaussian beam given the radius and the distance
    from the axis.

    Args:
    --
        power:      Beam Power in W
        radius:     1/e^2 radius, for an elliptical beam give the geometric mean.At focus, it's the waist
        radial_pos: radial distance from the center axis of the beam. Defaults to 0.
        omega_transition: angular fequency of the non-shifted transition

    Returns:
    --
        I: Intensity

    """
    I_max = 2 * power / pi / radius**2
    return I_max * np.exp(-2 * radial_pos**2 / radius**2)


def power2U(
    power,
    waist,
    wavelen=1064e-9,
    omega_transition=omega0,
    Er=None,
):
    """Returns the trap depth for a single beam given power and waist
    Parameters
    =========
    power               Beam Power in W
    waist               waist, for an elliptical beam give the geometric mean
    omega_transition    angular fequency of the non-shifted transition
    Er                  if None returns U in SI units
                        if 'Hz' return in 2pi Hz
                        if a number returns U in units of recoil energy.
    """
    omega = 2 * pi / wavelen
    U_SI = (
        -3
        * pi
        * c**2
        / (2 * omega_transition**3)
        * (
            Gamma / (omega_transition - omega)
            + Gamma / (omega_transition + omega)
        )
        * intensity(
            power,
            0,
            waist,
            omega_transition=omega_transition,
        )
    )
    if Er is None:
        return U_SI
    elif Er == "Hz":
        return U_SI / hplanck
    else:
        return U_SI / Er


# def intensity(
#     power,
#     radial_pos,
#     waist,
#     omega_transition=omega0,
# ):
#     """Returns the intensity profile of a gaussian beam given the waist
#     Parameters
#     =========
#     power               Beam Power in W
#     radial_pos          radial distance from the center axis of the beam
#     waist               waist, for an elliptical beam give the geometric mean
#     omega_transition    angular fequency of the non-shifted transition

#     """
#     I_max = 2 * power / pi / waist**2
#     return I_max * np.exp(-2 * radial_pos**2 / waist**2)
