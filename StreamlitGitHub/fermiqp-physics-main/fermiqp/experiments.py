from .lattice import GaussianBeam3D
import numpy as np
from scipy import constants as const
from pint import UnitRegistry

u = UnitRegistry()


class GaussianTrap:
    """Optical dipole trap created by a single beam

    """

    def __init__(self,power=1.0*u.W,waist=50*u.um,wavelen=1.064*u.um):
        
        if type(waist) is list:
            waist = np.array([w.to(u.m).magnitude for w in waist]) * u.m
        self.waist = waist
        self.wavelen = wavelen
        self.zR = np.pi * self.waist**2 / self.wavelen
        self.power = power
        if len(waist) == 2:
            self.waisty = waist[1]
            self.intensity = 2 * power / (np.pi * self.waist**2)



class PinningLattice:

    def __init__(self, waist=60e-6, amplitudes: list = [1, 1, 1, 1, 0, 0], phases: list = [0, 0, 0, 0, 0, 0], theta: float = np.pi/4, focuspos: np.ndarray = np.array([0, 0, 0])):
        ### amplitudes is a list of amplitudes for the beams: [1, 2s, 3s, 4, 2p, 3p]. First 4 are s-polarized components, last 2 are p-polarized components;

        if type(focuspos) is not np.ndarray:
            focuspos = np.array(focuspos)
        B1 = GaussianBeam3D(waist, focuspos=focuspos, prop_dir=np.array([np.cos(theta), np.sin(theta), 0]), amplitude=amplitudes[0], phi=phases[0])
        B2 = GaussianBeam3D(waist, focuspos=focuspos, prop_dir=np.array([-np.cos(theta), np.sin(theta), 0]),amplitude=amplitudes[1], phi=phases[1])
        B3 = GaussianBeam3D(waist, focuspos=focuspos, prop_dir=np.array([np.cos(theta), -np.sin(theta), 0]),amplitude=amplitudes[2], phi=phases[2])
        B4 = GaussianBeam3D(waist, focuspos=focuspos, prop_dir=np.array([-np.cos(theta), -np.sin(theta), 0]),amplitude=amplitudes[3], phi=phases[3])
        B5 = GaussianBeam3D(waist, focuspos=focuspos, prop_dir=np.array([-np.cos(theta), np.sin(theta), 0]),amplitude=amplitudes[4], phi=phases[4])
        B6 = GaussianBeam3D(waist, focuspos=focuspos, prop_dir=np.array([np.cos(theta), -np.sin(theta), 0]),amplitude=amplitudes[5], phi=phases[5])
        self.beams_s = [B1, B2, B3, B4]
        self.beams_p = [B5, B6]
    
    def scalar_field(self, X, Y, Z=0, pol='s'):
        if pol == 's':
            return sum([b.field([X, Y, Z]) for b in self.beams_s])
        elif pol == 'p':
            return sum([b.field([X, Y, Z]) for b in self.beams_p])

    def potential(self, X, Y, Z=0):
        return np.abs(self.scalar_field(X, Y, Z, pol='s'))**2 + np.abs(self.scalar_field(X, Y, Z, pol='p'))**2





class GeometryBeam(PinningLattice):

    def __init__(self, waist=60e-6, amplitudes:float = [1, 1], theta: float = np.pi/4,focuspos: np.ndarray = np.array([0, 0, 0]), phase: float = 0, wavelen: float = 1064e-9):
        if type(focuspos) is not np.ndarray:
            focuspos = np.array(focuspos)
        B1 = GaussianBeam3D(waist, wavelen=wavelen, focuspos=focuspos, prop_dir=np.array([np.cos(theta), -np.sin(theta), 0]),amplitude=amplitudes[0])
        B2 = GaussianBeam3D(waist, wavelen=wavelen, focuspos=focuspos, prop_dir=np.array([-np.cos(theta), np.sin(theta), 0]),amplitude=amplitudes[1], phi=phase)
        self.beams = [B1, B2]
    
    def scalar_field(self, X, Y, Z=0, pol='s'):
        return sum([b.field([X, Y, Z]) for b in self.beams])

    def potential(self, X, Y, Z=0):
        return np.abs(self.scalar_field(X, Y, Z))**2




