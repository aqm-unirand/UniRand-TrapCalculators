from importlib import resources
import numpy as np
from functools import lru_cache

DATA_DIR = resources.files('fermiqp.datas')

@lru_cache()
def load_zeeman(state='2S'):
    """ load Zeeman shift data for a state in 2S, 2P, 3S, 3P of Lithium 6, return B, energy"""
    with (DATA_DIR.joinpath("zeeman_"+state+'.npz')).open('rb') as f:
        return np.load(f)['B'], np.load(f)['energy']

@lru_cache()
def load_scatteringlengths():
    """ load scattering lengths data for Lithium 6, return B, a_scattering[1-2, 1-3, 2-3]"""
    datas = np.loadtxt(DATA_DIR.joinpath("scatteringlengths.txt"))

    return datas[:, 0], datas[:, 1:]


@lru_cache()
def load_lattice1d():
    """ load 1D lattice data for tunneling and interaction vs lattice depth, return V_vals, t_vals, wint_vals"""
    with (DATA_DIR.joinpath("lattice1d.npz")).open('rb') as f:
        data = np.load(f)
        return data['V_vals'], data['t_vals'], data['wint_vals']
    
@lru_cache()
def load_lattice2d():
    """ load 2D lattice data for tunneling and interaction vs lattice depth, return V_vals, t_vals, wint_vals"""
    with (DATA_DIR.joinpath("lattice2d.npz")).open('rb') as f:
        data = np.load(f)
        return data['V_vals'], data['t_vals'], data['wint_vals']