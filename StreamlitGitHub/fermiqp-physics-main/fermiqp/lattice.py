from re import L
import numpy as np
from math import pi
import warnings
import matplotlib.pyplot as plt
from scipy.linalg import eigh


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
    U = -(
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
            U = abs(U)
        elif unit == "K":
            U = temp2freq(abs(U))

        if axis == "radial":
            try:
                self.waisty
            except Exception:
                frequency = (4 * U * hplanck / mass / self.waist**2) ** 0.5 / 2 / pi
            else:
                frequency = [
                    (4 * U * hplanck / mass / self.waist**2) ** 0.5 / 2 / pi,
                    (4 * U * hplanck / mass / self.waisty**2) ** 0.5 / 2 / pi,
                ]
        elif axis == "axial":
            try:
                self.waisty
            except Exception:
                frequency = (2 * U * hplanck / mass / self.zR**2) ** 0.5 / 2 / pi
            else:
                # Use expression derived in Lucas mathematice notebook
                wz0 = (2 * pi * self.waist**2 * self.waisty**2) / (
                    self.wavelen * np.sqrt(self.waist**4 + self.waisty**4)
                )
                frequency = (4 * U * hplanck / mass / wz0**2) ** 0.5 / 2 / pi
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
                np.sqrt(hbar / mass / self.trap_freq(U=U, axis="radial") / 2 / pi),
                np.sqrt(hbar / mass / self.trap_freq(U=U, axis="axial") / 2 / pi),
            ]
        else:
            a_ho = [
                np.sqrt(hbar / mass / self.trap_freq(U=U, axis="radial")[0] / 2 / pi),
                np.sqrt(hbar / mass / self.trap_freq(U=U, axis="radial")[1] / 2 / pi),
                np.sqrt(hbar / mass / self.trap_freq(U=U, axis="axial") / 2 / pi),
            ]

        return a_ho


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

        return 1 / 2 / self.a * np.sqrt(2 * abs(U) * hplanck / mass)

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
        # self.U = Udd
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

        return 2 * np.sqrt(U / self.Er) * self.Er

    def a_ho(self, U=None):
        if U is None:
            U = self.U

        return np.sqrt(hbar / mass / self.on_site_freq(U=U) / 2 / pi)

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
        phi:         Arbitrary Phase in rad. NOTE: this translates in a 2*phi when
                        calculating the intensity
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
        waist,
        # waisty=None,
        focuspos=np.array([0, 0, 0]),
        wavelen=1.064e-6,
        prop_dir=np.array([1, 0, 0]),
        polar_dir=np.array([0, 0, 1]),
        phi=0.0,
        power=None,  # W
        depth=None,
        amplitude=None,  # amplitude of the field in a.u.
    ) -> None:
        """
        Parameters:
            waist:       Gaussian beam waist at the focu in m
            waisty:      second beam at waist
            focuspos:    Positon of the focusm, Defaults to [0,0,0]
            wavelen:     Wavelength in m. Defaults to 1064e-9
            prop_dir:    Propagation direction
            polar_dir:   Polarization direction (linear)
            power:       Optional, The power in each individual beam in W
            depth:       Optional, Depth of the dipole trap crated by the beam in Hz.
                            NOTE: There are no checks on physical meanings of values

        """

        self.waist = waist
        # self.waist = waist
        self.wavelen = wavelen

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
        self.__setPowerDepth(power, depth)
        self.amplitude = amplitude

        if self.waist <= self.wavelen:
            raise warnings.warn("Paraxial approximation no longer valid")
        # print(self.wavelen, self.power, self.depth)

    def __setPowerDepth(self, power, depth):
        # print(power, depth)
        self.depth = depth
        self.power = power
        if power is not None and depth is not None:
            raise ValueError("Give either Power or depth, not both")
        if depth is None and power is not None:
            self.depth = power2freq(power, waist=self.waist, wavelen=self.wavelen)
            self.power = power
        if depth is not None and power is None:
            # depth is linear in power
            self.power = np.abs(
                depth / power2freq(power=1, waist=self.waist, wavelen=self.wavelen)
            )
            self.depth = depth
        # return

    def getWaist(self, pos):
        """
        Calculates the waist in a plane given by the parameter pos
        Parameters:
        -----------
        pos:    np.array or float. if float, it's the distance from the focus along the propagation direction.
        along the propagation direction. If array, it's a point in 3D space

        Returns:
        -----------
        waistz_x, waistz_y    major and minor waists at position pos
        """
        if isinstance(pos, list):
            pos = np.array(pos)
        if isinstance(pos, np.ndarray) and pos.shape[0] == 3:
            z = (self.focuspos - pos) @ self.prop_dir
        else:
            z = pos
        # else:
        # raise ValueError("enter a valid position")

        waist_z = self.waist * np.sqrt(1 + z**2 / self.zR**2)

        return waist_z

    def orderPosition(self, pos):
        """
        A function that (hopefully) takes pos and returns an array of the
        positions along the propagation direction and one along the radial directions.
        """

        if len(pos) == 2:
            X, Y = pos
            Z = 0
        else:
            X, Y, Z = pos

        # distance from Focus
        R = ((X - self.x0) ** 2 + (Y - self.y0) ** 2 + (Z - self.z0) ** 2) ** 0.5

        # distance along the axial direction:
        z_axial = (
            self.prop_dir[0] * (X - self.x0)
            + self.prop_dir[1] * (Y - self.y0)
            + self.prop_dir[2] * (Z - self.z0)
        )

        # Distance along the radial direction
        r = np.sqrt(np.abs(R**2 - z_axial**2))

        # if isinstance(self, EllipticalGaussianBeam):
        #     return z_axial,
        return z_axial, r

    def field(self, pos, amplitude=None):
        """
        Calculates the Electric field in a.u. of a gaussian beam with circular
        cross section in far field approximation

        Parameters:
        ----------
        pos:    array or list [X,Y] or [X,Y,Z]
                    coordinates in 3D space
                if pos=[X,Y] field will have shape (len(Y), len(X))
                X,Y should be meshgrids


        Returns:
        ----------
        vector_field:   list o f length 3 containing Ex, Ey, Ez, arrays of the
                        same shape as pos.

        Accounts for wavefront curvature and Gouy phase
        NOTE: only valid in in far field approximation (kz>>1) and far from the
        diffraction limit (kw>>1)
        """

        # warnings.warn("E-field calculation doesn't account for power")

        if len(pos) == 2:
            X, Y = pos
            Z = 0
        else:
            X, Y, Z = pos

        if amplitude is None:
            if self.amplitude is None:
                amplitude = 1.0  # a.u.
            else:
                amplitude = self.amplitude

        # distance from the focus position:
        R = ((X - self.x0) ** 2 + (Y - self.y0) ** 2 + (Z - self.z0) ** 2) ** 0.5

        # distance along the axial direction:
        z_axial = (
            self.prop_dir[0] * (X - self.x0)
            + self.prop_dir[1] * (Y - self.y0)
            + self.prop_dir[2] * (Z - self.z0)
        )

        # Distance along the radial direction
        r = np.sqrt(np.abs(R**2 - z_axial**2))

        # waisty = self.waisty * np.sqrt(1 + z_axial**2 / self.zRy**2)
        # waistx = self.waistx * np.sqrt(1 + z_axial**2 / self.zRx**2)
        waist = self.waist * np.sqrt(1 + z_axial**2 / self.zR**2)

        # wavefront_curv_factor_x = z_axial / (z_axial + self.zRx)  # =1/curvature
        # wavefront_curv_factor_y = z_axial / (z_axial + self.zRy)  # =1/curvature
        wavefront_curv_factor = z_axial / (z_axial + self.zR)  # =1/curvature

        gouy_phase = np.arctan(z_axial / self.zR)

        # return field
        # NOTE: far field approximation

        scalar_field = amplitude * (
            self.waist
            / waist
            * np.exp(-(r**2) / waist**2)  # Gaussian profile
            * np.exp(
                -1j
                * (
                    X * self.prop_dir[0] + Y * self.prop_dir[1] + Z * self.prop_dir[2]
                )  # k x
                * self.k
                + 1j * gouy_phase
                - 1j * self.phi
            )
            * np.exp(-1j * self.k * r**2 * wavefront_curv_factor / 2)
        )

        return scalar_field

    def getVectorField(self, pos):
        """
        Returns:
        ----------
        vector_field:   list o f length 3 containing Ex, Ey, Ez, arrays of the
                        same shape as pos.
        """
        scalar_field = self.field(pos)
        vector_field = [self.polar_dir[i] * scalar_field for i in range(3)]
        return np.array(vector_field)

    def getIntensity(self, pos):
        """Calculates the intensity of the beam, accounting for power.

        Parameters:
            pos:    array of the form [X,Y,Z]

        """

        E_vec = self.getVectorField(pos)
        E_amplitude_sq = (E_vec.conj() * E_vec).real.sum(axis=0)

        Imax = 2 * self.power / pi / self.waist**2

        return E_amplitude_sq * Imax
        # return E_amplitude_sq

        # intensity = np.zeros_like(pos)


class InterferingBeams:
    """Constructs interference patterns from gaussian beams

    Parameters:
    -------------
    beams:   list or array of beam

    Look at examples\9-interfering_beams.ipynb for usage example

    """

    def __init__(self, beams):
        # if type(beams) != list and not (isinstance(beams, GaussianBeam3D)):
        if not (isinstance(beams, list)) and not (isinstance(beams, GaussianBeam3D)):
            raise Warning("Beams need to be a GaussianBeam3D or a list of them")
        if isinstance(beams, list):
            for element in beams:
                if not isinstance(element, GaussianBeam3D):
                    raise TypeError("Each entry must be an instance of GaussianBeam3D")
                    # elif not (isinstance(beams, GaussianBeam3D)):
                    # for el in beams:
                    #     print(type(el))
                    #     if not (isinstance(el, GaussianBeam3D)) or not (
                    #         isinstance(el, EllipticalGaussianBeam)
                    #     ):
                    raise TypeError("Each entry must be an instance of GaussianBeam3D")
        self.beams = beams

    def getIntensity(self, pos):
        """
        NOTE: there is no time averaging. Beams of wildly different wavelengths
        will still create an interference pattern.

        """
        field = np.zeros_like(self.beams[0].getVectorField(pos))
        for beam in self.beams:
            f = beam.getVectorField(pos)
            Imax = 2 * beam.power / pi / beam.waist**2
            field += f * np.sqrt(Imax)
            print(Imax)

        intensity = (field.conj() * field).real.sum(axis=0)

        return intensity

    def getPotential(self, pos):
        """
        Assumes all the beams are far detuned
        Returns the potential (with sign) of TLS
        """
        field = np.zeros_like(self.beams[0].getVectorField(pos))

        for beam in self.beams:
            f = beam.getVectorField(pos)
            u = power2freq(beam.power, waist=beam.waist, wavelen=beam.wavelen)
            field += f * np.sqrt(np.abs(u))

        pot = (field.conj() * field).real.sum(axis=0) * np.sign(u)

        return pot


class EllipticalGaussianBeam(GaussianBeam3D):
    def __init__(
        self,
        waistx,
        waisty=None,
        focuspos=np.array([0, 0, 0]),
        wavelen=1.064e-6,
        prop_dir=np.array([1, 0, 0]),
        polar_dir=np.array([0, 0, 1]),
        phi=0.0,
        power=None,  # W
        depth=None,
        angle=0,  # deg
        wx_axis=np.array([0, 1, 0]),
    ) -> None:
        """
        Most of the parameters are the same as on GaussiaBwam3D
        Parameters:
        ---------------
            waistx:      Gaussian beam waist at the focus
            waisty:      beam waist
            focuspos:    Positon of the focusm, Defaults to [0,0,0]
            wavelen:     Wavelength in m. Defaults to 1064e-9
            prop_dir:    Propagation direction
            polar_dir:   Polarization direction (linear)
            power:       Optional, The power in each individual beam in W
            depth:       Optional, Depth of the dipole trap crated by the beam in Hz.
                            NOTE: There are no checks on physical meanings of values
            wx_axis:        direction of the waistx. Needs to be a 3D vector in space

        """
        self.waisty = waisty
        self.waistx = waistx
        self.waist_geom = np.sqrt(self.waistx * self.waisty)
        self.waist = self.waist_geom
        self.angle = angle
        self.wx_axis = wx_axis / np.linalg.norm(wx_axis)

        super().__init__(
            self.waist_geom, focuspos, wavelen, prop_dir, polar_dir, phi, power, depth
        )

        # Two ciruclar gaussian beams of waists waistx and waisty (can be useful)
        self.beamx = GaussianBeam3D(
            waistx, focuspos, wavelen, prop_dir, polar_dir, phi, power=self.power
        )
        self.beamy = GaussianBeam3D(
            waisty, focuspos, wavelen, prop_dir, polar_dir, phi, power=self.power
        )

        if self.waist_geom <= self.wavelen:
            raise warnings.warn("Paraxial approximation no longer valid")

        self.zRx = pi * self.waistx**2 / self.wavelen
        self.zRy = pi * self.waisty**2 / self.wavelen

        if np.abs(self.prop_dir @ self.wx_axis) > 1e-15:
            raise ValueError("axis needs to be perpendicular to propagation directoin")

    def getWaist(self, axial):
        """
        Parameters:
        -------------
        axial:          array or float, distance from focus along propagation direction

        """
        waistx_z = self.beamx.getWaist(axial)
        waisty_z = self.beamy.getWaist(axial)

        return waistx_z, waisty_z

    def getScalarField(self, pos):
        """
        Calculates scalar field
        """
        return self.field(pos)

    def beamCoords(self, pos):
        """
        A function that that converts position in cartesian 3D space
        to beam coordinates (distance along the propagation axis,
        distance along minor and major axis)


        Returns
        -----------
        axial:      z, distance from the focus along the propagation axis
        axis_x:     x, distance from the propagation axis, projected on the waistx direction
        axis_y:     y, distance from the propagation axis, projected on the waisty direction


        note: axis_x**2 + axis_y**2 = r**2
        """

        if len(pos) == 2:
            X, Y = pos
            Z = 0
        else:
            X, Y, Z = pos

        ######## distance from Focus

        Rx = X - self.x0
        Ry = Y - self.y0
        Rz = Z - self.z0
        Rvec = [Rx, Ry, Rz]

        R = list(map(lambda x: x**2, Rvec))
        R = sum(R) ** 0.5

        ######## distance along the axial direction:
        axial = sum([self.prop_dir[i] * Rvec[i] for i in range(3)])

        ######## Distance along the radial direction
        rvec = [Rvec[i] - self.prop_dir[i] * axial for i in range(3)]

        ########x and y coordinates
        axis_x = sum([rvec[i] * self.wx_axis[i] for i in range(3)])
        axis_y = np.sqrt(np.abs(sum(list(map(lambda x: x**2, rvec))) - axis_x**2))
        return axial, axis_x, axis_y  # , r

    def _1DGauss(self, x, z, whichwaist):
        """
        calculates the 1D gaussian distribution of the Electric field along
        the direction of waist x or waist y

        x:      radial coordinate
        z:      axial coordinate
        whichwaist: 0 for waistx, 1 for waisty

        """

        if whichwaist == 0:
            w0 = self.waistx
            zR = self.zRx
        elif whichwaist == 1:
            w0 = self.waisty
            zR = self.zRy
        else:
            raise ValueError

        q = np.array(zR - 1j * z)
        w = self.getWaist(z)[whichwaist]
        gouy = self.getGouy(z)[whichwaist]

        return (
            np.sqrt(w0 / w)
            * np.exp(-1j * self.k * z / 2 * (1 + x**2 / q.conj() / q))
            * np.exp(1j * gouy - 1j * self.phi)
            * np.exp(-(x**2) / w**2)
        )

    def field(self, pos):
        """
        Calculates the field along the x and y and multiplies them

        Parameters:
        --------------
        pos:        list or array, (meshgrid) of shape (3,)
        """

        z, x, y = self.beamCoords(pos)
        fieldx = self._1DGauss(x, z, 0)
        fieldy = self._1DGauss(y, z, 1)
        # prefactor = np.sqrt(2 * self.power * pi) * self.waist_geom / self.wavelen
        # prefactor = 1
        return fieldx * fieldy

    def getVectorField(self, pos):
        """
        Separates the components of the scalar field in the polarization components.


        Parameters:
        --------------
        pos:        list or array, (meshgrid) of shape (3,)


        Returns:
        vector_field      np.array of shape (3,...)

        TODO: should, in principle be more subtle if the polarization has a component
        along the propagation direction?

        """
        scalar_field = self.field(pos)
        vector_field = [self.polar_dir[i] * scalar_field for i in range(3)]
        return np.array(vector_field)

    def getGouy(self, z):
        """
        Computes the gouy phase.


        Parameters:
        -------------
        axial:          array or float, distance from focus along propagation direction

        """
        return np.arctan(z / self.zRx), np.arctan(z / self.zRy)

    def qParameter(self, z):
        """
        Computes the gaussian beam complex parameter.


        Parameters:
        -------------
        z:          array or float, distance from focus along propagation direction


        Returns:
        -------------
        Tuple of the complex parameter along x and y

        """
        return z - 1j * self.zRx, z - 1j * self.zRy


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
        * (Gamma / (omega_transition - omega) + Gamma / (omega_transition + omega))
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


class Lattice1D:
    def __init__(self, V, m_max=10, q_num=10, x_num=200, x_max=2):
        self.V = V
        self.m_max = m_max
        self.q_num = q_num

        self.q_vals = np.arange(2*self.q_num)/(self.q_num) - 1 ## in unit of pi
        self.m_vals = np.arange(-self.m_max, self.m_max+1)

        self.x_vals = np.arange(x_num)/x_num * x_max * 2 - x_max

        self.bloch_waves = None
        self.wannier_waves = None
    
    def fourier_component(self, delta_m):
        if delta_m == 0:
            return -self.V / 2
        elif np.abs(delta_m) == 1:
            return -self.V / 4
        else:
            return 0

    
    def Hamiltonian(self, q):
        H = np.zeros((2*self.m_max+1, 2*self.m_max+1), dtype=complex)
        for i, m in enumerate(self.m_vals):
            H[i,i] = (2*m+q)**2
            for j, m in enumerate(self.m_vals):
                if i != j:
                    H[i,j] += self.fourier_component(i-j)

        return H

    def get_bloch(self):
        
        self.energy = np.zeros((len(self.m_vals), len(self.q_vals)))
        self.bloch_waves = np.zeros((len(self.m_vals), len(self.q_vals), len(self.x_vals)), dtype=complex)

        for i, q in enumerate(self.q_vals):
            H = self.Hamiltonian(q)

            evals, evecs = eigh(H)
            self.energy[:, i] = evals
            blochbasis = np.array([np.exp(1j*np.pi*(q+2*self.m_vals[im])*self.x_vals) for im in range(len(self.m_vals))])

            self.bloch_waves[:, i, :] = evecs.swapaxes(0, 1) @ blochbasis
        
    def get_wannier(self, band_index=[0]):

        if self.bloch_waves is None:
            self.get_bloch()

        x0 = 0

        self.wannier_waves = []
        for ib in band_index:

            if np.mod(ib, 2) == 0:

                w = (np.sum(self.bloch_waves[ib,:,:] * np.exp(-1j*np.angle(self.bloch_waves[ib, :, np.argmin(np.abs(self.x_vals-x0))]))[:, None], axis=0))
            
            else:
                pdv = np.diff(self.bloch_waves[ib, :, :], axis=-1)

                xi = np.argmin(np.abs(self.x_vals-x0))

                theta = np.angle((pdv[:, xi]+pdv[:, xi+1])/2)

                w = (np.sum(self.bloch_waves[ib,:,:] * np.exp(-1j*theta[:, None]), axis=0))

            # w = (np.sum(self.bloch_waves[ib,:,:] * np.exp(-1j*np.angle(self.bloch_waves[ib, :, np.argmin(np.abs(self.x_vals-x0))]))[:, None], axis=0))
            # w = w/np.sqrt(np.sum(np.abs(w)**2))
            # w = w/(np.trapz(np.abs(w)**2, x=self.x_vals))**0.5

            w = w/(np.trapezoid(np.abs(w)**2, x=self.x_vals))**0.5

            self.wannier_waves.append(w)

        # self.wannier_waves = np.array(self.wannier_waves)

        return self.wannier_waves




class Superlattice1D(Lattice1D):

    def __init__(self, Vs, Vl=1e-2, phi=0, m_max=10, q_num=10, x_num=200, x_max=2):
        self.Vs = Vs
        self.Vl = Vl
        self.phi = phi
        self.m_max = m_max
        self.q_num = q_num

        self.q_vals = np.arange(2*self.q_num)/(self.q_num) - 1 ## in unit of pi
        self.m_vals = np.arange(-self.m_max, self.m_max+1)

        self.x_vals = np.arange(x_num)/x_num * x_max * 2 - x_max

        self.s = 0.5 ## Band mixing parameter

        self.bloch_waves = None


    def fourier_component(self, delta_m):
        # if delta_m == 0:
        #     return -self.V / 2
        # elif np.abs(delta_m) == 1:
        #     return -self.V / 4
        # else:
        #     return 0
        
        if np.abs(delta_m) == 0:
            V = 0.5 * self.Vs - 0.5 * self.Vl
        elif delta_m == 1:
            V = -self.Vl / 4 * np.exp(1j*self.phi*np.pi)
        elif delta_m == -1:
            V = -self.Vl / 4 * np.exp(-1j*self.phi*np.pi)
        elif (delta_m) == 2:
            V = self.Vs / 4
        elif (delta_m) == -2:
            V = self.Vs / 4
        else:
            V = 0
        return V
    
    def get_wannier(self, band_index=[0]):

        if self.bloch_waves is None:
            self.get_bloch()

        xL = -0.25
        xR = 0.25

        xLi = np.argmin(np.abs(self.x_vals-xL))
        xRi = np.argmin(np.abs(self.x_vals-xR))

        self.wannier_waves = []

        ib = 0
        wL0 = (np.sum(self.bloch_waves[ib,:,:] * np.exp(-1j*np.angle(self.bloch_waves[ib, :, xLi]))[:, None], axis=0))
        wR0 = (np.sum(self.bloch_waves[ib,:,:] * np.exp(-1j*np.angle(self.bloch_waves[ib, :, xRi]))[:, None], axis=0))
        
        ib = 1

        wL1 = (np.sum(self.bloch_waves[ib,:,:] * np.exp(-1j*np.angle(self.bloch_waves[ib, :, xLi]))[:, None], axis=0))
        wR1 = (np.sum(self.bloch_waves[ib,:,:] * np.exp(-1j*np.angle(self.bloch_waves[ib, :, xRi]))[:, None], axis=0))


        self.wL0 = wL0
        self.wL1 = wL1
        self.wR0 = wR0
        self.wR1 = wR1

        self.fit_s()

        wL, wR = self.band_mixing()

        self.wannier_waves = [wL, wR]

        return wL, wR
    

    def band_mixing(self, s=None):
        if s is None:
            s = self.s
        
        wL = s**0.5 * self.wL0 + (1-s)**0.5 * self.wL1
        wR = (1-s)**0.5 * self.wR0 + s**0.5 * self.wR1

        return wL, wR
    


    def fit_s(self):

        s_vals = np.linspace(1e-2, 1-1e-2, 1000)

        self.s_vals = s_vals
        self.costs = []

        for s in s_vals:
            wL, wR = self.band_mixing(s)

            Delta_J_L = np.trapezoid(wL.conj()*wR*np.abs(wL)**2, x=self.x_vals)
            Delta_J_R = np.trapezoid(wR.conj()*wL*np.abs(wR)**2, x=self.x_vals)
            U_LR = np.trapezoid(np.abs(wL)**2*np.abs(wR)**2, x=self.x_vals)

            self.costs.append((np.abs(Delta_J_L)**2 + np.abs(Delta_J_R)**2 + np.abs(U_LR)**2))

        self.costs = np.array(self.costs)
        self.s = self.s_vals[np.argmin(self.costs)]

    def potential(self):

        return self.Vs * np.cos(2*np.pi * self.x_vals)**2 - self.Vl * np.cos(self.x_vals * np.pi + self.phi)**2




def meshflatten(x):
    X, Y = np.meshgrid(x, x)

    return X.flatten(), Y.flatten()


class Lattice2D(Superlattice1D):

    def __init__(self, V, theta=0, Vg=0, phi=0, m_max=5, q_num=5, r_num=200, r_max=2):
        self.V = V
        self.theta = theta
        self.Vg = Vg
        self.phi = phi

        self.m_max = m_max
        self.m_vals = np.arange(-self.m_max, self.m_max+1)
        self.mx_vals, self.my_vals = meshflatten(self.m_vals)
        
        self.q_num = q_num
        self.q_vals = np.arange(2*self.q_num)/(self.q_num) - 1
        self.qx_vals, self.qy_vals = meshflatten(self.q_vals)

        self.r_vals = np.arange(r_num)/r_num * r_max * 2 - r_max
        self.X, self.Y = np.meshgrid(self.r_vals, self.r_vals)
        self.x_vals, self.y_vals = meshflatten(self.r_vals)
        self.r_num = r_num

        self.s = 0.5 ## Band mixing parameter

        self.bloch_waves = None
        self.wannier_waves = None

    
    def fourier_component(self, delta_mx, delta_my):

        if (delta_mx) == 1 and np.abs(delta_my) == 0:
            V = self.V * np.sin(self.theta)/2j
        elif (delta_mx) == -1 and np.abs(delta_my) == 0:
            V = -self.V * np.sin(self.theta)/2j
        elif np.abs(delta_mx) == 0 and (delta_my) == 1:
            V = self.V * np.sin(self.theta)/2j
        elif np.abs(delta_mx) == 0 and (delta_my) == -1:
            V = -self.V * np.sin(self.theta)/2j
        elif np.abs(delta_mx) == 0 and np.abs(delta_my) == 0:
            V = -self.V - self.Vg/2
        elif np.abs(delta_mx) == 1 and np.abs(delta_my) == 1:
            if delta_mx == delta_my:
                V = self.V / 4
            else:
                V =  - self.V / 4 + self.Vg/4 * np.exp(1j*self.phi*np.pi)
        else:
            V = 0
        return V




    def Hamiltonian(self, qx, qy):
        H = np.zeros((len(self.mx_vals), len(self.mx_vals)), dtype=complex)
        for i in range(len(self.mx_vals)):
            mxi, myi = self.mx_vals[i], self.my_vals[i]
            H[i, i] = (2*mxi+qx)**2+(2*myi+qy)**2
            for j in range(len(self.mx_vals)):
                mxj, myj = self.mx_vals[j], self.my_vals[j]
                H[i, j] += self.fourier_component(mxi-mxj, myi-myj)
        return H
    

    def get_bloch(self):

        self.energy = np.zeros((len(self.mx_vals), len(self.qx_vals)))
        self.bloch_waves = np.zeros((len(self.mx_vals), len(self.qx_vals), len(self.x_vals)), dtype=complex)


        for i in range(len(self.qx_vals)):
            qx, qy = self.qx_vals[i], self.qy_vals[i]
            H = self.Hamiltonian(qx, qy)
            evals, evecs = eigh(H)
            self.energy[:, i] = evals

            blochbasis = np.array([np.exp(1j*np.pi*2*(qx/2+self.mx_vals[im])*self.x_vals + 1j*np.pi*2*(qy/2+self.my_vals[im])*self.y_vals) for im in range(len(self.mx_vals))])

            self.bloch_waves[:, i, :] = evecs.swapaxes(0, 1) @ blochbasis
        

    def get_wannier(self, rL=(-0.25, -0.25), rR=(0.25, 0.25), s=None):

        if self.bloch_waves is None:
            self.get_bloch()

        xL, yL = rL
        xR, yR = rR

        rLi = np.argmin(np.abs(self.x_vals-xL)+np.abs(self.y_vals-yL))
        rRi = np.argmin(np.abs(self.x_vals-xR)+np.abs(self.y_vals-yR))


        

        self.wannier_waves = []

        ib = 0
        self.thetaL0 = np.angle(self.bloch_waves[ib, :, rLi])
        self.thetaR0 = np.angle(self.bloch_waves[ib, :, rRi])
        wL0 = (np.sum(self.bloch_waves[ib,:,:] * np.exp(-1j*self.thetaL0)[:, None], axis=0))
        wR0 = (np.sum(self.bloch_waves[ib,:,:] * np.exp(-1j*self.thetaR0)[:, None], axis=0))

        ib = 1
        self.thetaL1 = np.angle(self.bloch_waves[ib, :, rLi])
        self.thetaR1 = np.angle(self.bloch_waves[ib, :, rRi])

        wL1 = (np.sum(self.bloch_waves[ib,:,:] * np.exp(-1j*self.thetaL1)[:, None], axis=0))
        wR1 = (np.sum(self.bloch_waves[ib,:,:] * np.exp(-1j*self.thetaR1)[:, None], axis=0))


        self.wL0 = wL0
        self.wL1 = wL1
        self.wR0 = wR0
        self.wR1 = wR1

        if s is None:
            self.fit_s()
        
        else:
            self.s = s

        wL, wR = self.band_mixing()

        self.wannier_waves = [wL, wR]

        return wL, wR
    

    def band_mixing(self, s=None):
        if s is None:
            s = self.s
        
        wL = s**0.5 * self.wL0 + (1-s)**0.5 * self.wL1
        wR = (1-s)**0.5 * self.wR0 + s**0.5 * self.wR1

        wL = wL.reshape((self.r_num, self.r_num))
        wR = wR.reshape((self.r_num, self.r_num))

        return wL, wR
    
    def fit_s(self):

        s_vals = np.linspace(0.5, 1, 2000)

        self.s_vals = s_vals
        self.costs = []

        for s in s_vals:
            wL, wR = self.band_mixing(s)

            Delta_J_L = np.sum(wL.conj()*wR*np.abs(wL)**2)
            Delta_J_R = np.sum(wR.conj()*wL*np.abs(wR)**2)
            U_LR = np.sum(np.abs(wL)**2*np.abs(wR)**2)

            self.costs.append((np.abs(Delta_J_L)**2 + np.abs(Delta_J_R)**2 + np.abs(U_LR)**2)**0.5)

        self.costs = np.array(self.costs)
        self.s = self.s_vals[np.argmin(self.costs)]

    
    def potential(self):

        V = - self.V *(np.sin(np.pi*(self.X+self.Y))**2 + np.cos(np.pi*(self.X-self.Y))**2 - 2 * np.sin(self.theta) * np.sin(np.pi*(self.X+self.Y)) * np.cos(np.pi*(self.X-self.Y))) - self.Vg * np.cos(np.pi*(self.X-self.Y)+self.phi)**2
        return V
    
    def tunneling(self):
        
        if self.wannier_waves is None:
            self.get_wannier()

        tLL = -np.sum(self.s * self.energy[0,:] * np.exp(1j*self.qx_vals*np.pi) + (1-self.s) * self.energy[1,:] * np.exp(1j*self.qx_vals*np.pi))/len(self.qx_vals)
        tLLp = -np.sum(self.s * self.energy[0,:] * np.exp(1j*(self.qx_vals*np.pi+self.qy_vals*np.pi)) + (1-self.s) * self.energy[1,:] * np.exp(1j*(self.qx_vals*np.pi+self.qy_vals*np.pi)))/len(self.qx_vals)
        tRR = -np.sum((1-self.s) * self.energy[0,:] * np.exp(1j*self.qx_vals*np.pi) + self.s * self.energy[1,:] * np.exp(1j*self.qx_vals*np.pi))/len(self.qx_vals)
        tLR = -np.sqrt(self.s*(1-self.s)) * np.sum(np.exp(1j*self.thetaL0) * np.exp(-1j*self.thetaR0)*self.energy[0,:] + np.exp(1j*self.thetaL1) * np.exp(-1j*self.thetaR1)*self.energy[1,:])/len(self.qx_vals)

        return tLL.real, tRR.real, tLLp.real, tLR.real

    

class LatticeEmery(Lattice2D):

    def __init__(self, V, theta=0, Vg=0, phi=0, m_max=5, q_num=5, r_num=200, r_max=2):
        self.V = V
        self.theta = theta
        self.Vg = Vg
        self.phi = phi

        self.m_max = m_max
        self.m_vals = np.arange(-self.m_max, self.m_max+1)
        self.mx_vals, self.my_vals = meshflatten(self.m_vals)
        
        self.q_num = q_num
        self.q_vals = np.arange(2*self.q_num)/(self.q_num) - 1
        self.qx_vals, self.qy_vals = meshflatten(self.q_vals)

        self.r_vals = np.arange(r_num)/r_num * r_max * 2 - r_max
        self.X, self.Y = np.meshgrid(self.r_vals, self.r_vals)
        self.x_vals, self.y_vals = meshflatten(self.r_vals)
        self.r_num = r_num

        self.s = 0.5 ## Band mixing parameter

        self.bloch_waves = None
        self.wannier_waves = None

    
    def fourier_component(self, delta_mx, delta_my):
        
        beta = np.sin(self.theta)
        V0 = -self.V

        Vl = beta * V0

        phi = -np.pi/2

        if (delta_mx) == 2 and np.abs(delta_my) == 0:
            V = V0 / 4j * np.exp(1j*phi)
        elif (delta_mx) == -2 and np.abs(delta_my) == 0:
            V = -V0 / 4j * np.exp(-1j*phi)
        elif np.abs(delta_mx) == 0 and (delta_my) == 2:
            V = V0 / 4j * np.exp(1j*phi)
        elif np.abs(delta_mx) == 0 and (delta_my) == -2:
            V = -V0 / 4j * np.exp(-1j*phi)
        elif np.abs(delta_mx) == 0 and np.abs(delta_my) == 0:
            V = V0 / 2
        elif np.abs(delta_mx) == 1 and np.abs(delta_my) == 1:
            if delta_mx == delta_my:
                V = -Vl/2
            else:
                V = Vl/2
        else:
            V = 0
            
        if np.mod(delta_mx + delta_my, 2) == 0:
            V -= 0.1 * V0 * (-1)**(np.abs(delta_mx + delta_my)//2)
        else:
            V -= 0.1j * V0 * (-1)**(np.abs(delta_mx + delta_my + 1)//2)
    

        return V




    def Hamiltonian(self, qx, qy):
        H = np.zeros((len(self.mx_vals), len(self.mx_vals)), dtype=complex)
        for i in range(len(self.mx_vals)):
            mxi, myi = self.mx_vals[i], self.my_vals[i]
            H[i, i] = (2*mxi+qx)**2+(2*myi+qy)**2
            for j in range(len(self.mx_vals)):
                mxj, myj = self.mx_vals[j], self.my_vals[j]
                H[i, j] += self.fourier_component(mxi-mxj, myi-myj)
        return H
    

    def get_bloch(self):

        self.energy = np.zeros((len(self.mx_vals), len(self.qx_vals)))
        self.bloch_waves = np.zeros((len(self.mx_vals), len(self.qx_vals), len(self.x_vals)), dtype=complex)


        for i in range(len(self.qx_vals)):
            qx, qy = self.qx_vals[i], self.qy_vals[i]
            H = self.Hamiltonian(qx, qy)
            evals, evecs = eigh(H)
            self.energy[:, i] = evals

            blochbasis = np.array([np.exp(1j*np.pi*2*(qx/2+self.mx_vals[im])*self.x_vals + 1j*np.pi*2*(qy/2+self.my_vals[im])*self.y_vals) for im in range(len(self.mx_vals))])

            self.bloch_waves[:, i, :] = evecs.swapaxes(0, 1) @ blochbasis
        

    def get_wannier(self, rd=(-0.25, -0.25), rpy=(-0.25, 0.25), rpx=(0.25, -0.25)):

        if self.bloch_waves is None:
            self.get_bloch()

        xd, yd = rd
        xpy, ypy = rpy
        xpx, ypx = rpx

        rdi = np.argmin(np.abs(self.x_vals-xd)+np.abs(self.y_vals-yd))
        rpyi = np.argmin(np.abs(self.x_vals-xpy)+np.abs(self.y_vals-ypy))
        rpxi = np.argmin(np.abs(self.x_vals-xpx)+np.abs(self.y_vals-ypx))


        self.wannier_waves = []

        ib = 0
        self.thetad0 = np.angle(self.bloch_waves[ib, :, rdi])
        self.thetapy0 = np.angle(self.bloch_waves[ib, :, rpyi])
        self.thetapx0 = np.angle(self.bloch_waves[ib, :, rpxi])
        wd0 = (np.sum(self.bloch_waves[ib,:,:] * np.exp(-1j*self.thetad0)[:, None], axis=0))
        wpy0 = (np.sum(self.bloch_waves[ib,:,:] * np.exp(-1j*self.thetapy0)[:, None], axis=0))
        wpx0 = (np.sum(self.bloch_waves[ib,:,:] * np.exp(-1j*self.thetapx0)[:, None], axis=0))

        ib = 1
        self.thetad1 = np.angle(self.bloch_waves[ib, :, rdi])
        self.thetapy1 = np.angle(self.bloch_waves[ib, :, rpyi])
        self.thetapx1 = np.angle(self.bloch_waves[ib, :, rpxi])

        wd1 = (np.sum(self.bloch_waves[ib,:,:] * np.exp(-1j*self.thetad1)[:, None], axis=0))
        wpy1 = (np.sum(self.bloch_waves[ib,:,:] * np.exp(-1j*self.thetapy1)[:, None], axis=0))
        wpx1 = (np.sum(self.bloch_waves[ib,:,:] * np.exp(-1j*self.thetapx1)[:, None], axis=0))

        ib = 2
        self.thetad2 = np.angle(self.bloch_waves[ib, :, rdi])
        self.thetapy2 = np.angle(self.bloch_waves[ib, :, rpyi])
        self.thetapx2 = np.angle(self.bloch_waves[ib, :, rpxi])

        wd2 = (np.sum(self.bloch_waves[ib,:,:] * np.exp(-1j*self.thetad2)[:, None], axis=0))
        wpy2 = (np.sum(self.bloch_waves[ib,:,:] * np.exp(-1j*self.thetapy2)[:, None], axis=0))
        wpx2 = (np.sum(self.bloch_waves[ib,:,:] * np.exp(-1j*self.thetapx2)[:, None], axis=0))

        self.wd0 = wd0
        self.wd1 = wd1
        self.wd2 = wd2
        self.wpy0 = wpy0
        self.wpy1 = wpy1
        self.wpy2 = wpy2
        self.wpx0 = wpx0
        self.wpx1 = wpx1
        self.wpx2 = wpx2

        self.fit_s()

        wd, wpx, wpy = self.band_mixing(self.s)

        self.wd = wd
        self.wpx = wpx
        self.wpy = wpy

        return wd, wpx, wpy

    def band_mixing(self, s=None):
        if s is None:
            s = self.s
        
        wd = s**0.5 * self.wd0 + ((1-s)/2)**0.5 * self.wd1 + ((1-s)/2)**0.5 * self.wd2
        wpy = (1-s)**0.5 * self.wpy0 + (s/2)**0.5 * self.wpy1 + (s/2)**0.5 * self.wpy2
        wpx = (1-s)**0.5 * self.wpx0 + (s/2)**0.5 * self.wpx1 + (s/2)**0.5 * self.wpx2

        wd = wd.reshape((self.r_num, self.r_num))
        wpy = wpy.reshape((self.r_num, self.r_num))
        wpx = wpx.reshape((self.r_num, self.r_num))

        return wd, wpx, wpy
    
    def fit_s(self, smin=0.5, smax=1):

        if self.wannier_waves is None:
            self.get_wannier()

        s_vals = np.linspace(smin, smax, 2000)

        self.s_vals = s_vals
        self.costs = []

        for s in s_vals:
            wd, wpx, wpy = self.band_mixing(s)

            U_pd = np.sum(np.abs(wd)**2*np.abs(wpy)**2) + np.sum(np.abs(wd)**2*np.abs(wpx)**2) 

            self.costs.append((np.abs(U_pd)**2)**0.5)

        self.costs = np.array(self.costs)
        self.s = self.s_vals[np.argmin(self.costs)]

           