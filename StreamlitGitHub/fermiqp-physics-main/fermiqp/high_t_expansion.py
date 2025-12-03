from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify

# atomic mass (kg)
m = 9.9883414e-27

# Define symbols
mu, U, T, t_x, t_y = sp.symbols("mu U T t_x t_y", real=True, positive=True)

# Define constants
kB = 1  # natural units, k_B = 1
h = 6.63e-34


# Define beta, zeta, w
beta = 1 / (kB * T)
zeta = sp.exp(beta * mu)
w = sp.exp(-beta * U)

# Partition function z0
z0 = 1 + 2 * zeta + zeta**2 * w

# Omega (grand potential per site)
term1 = sp.log(z0)
term2 = beta**2 * (t_x**2 + t_y**2) / z0**2
bracket = 2 * zeta * (1 + zeta**2 * w) + (4 * zeta**2 / (beta * U)) * (1 - w)
Omega = -(term1 + term2 * bracket) / beta

# Derivatives
n = -sp.diff(Omega, mu)  # Single-site density
s = -sp.diff(Omega, T)  # Entropy
p_d = sp.diff(Omega, U)  # Probability of double occupancy

# Convert symbolic expressions to functions
variables = (mu, U, T, t_x, t_y)
n_func = lambdify(variables, n, modules="numpy")
s_func = lambdify(variables, s, modules="numpy")
pd_func = lambdify(variables, p_d, modules="numpy")
ndet_func = lambdify(variables, n - 2 * p_d, modules="numpy")


def mu_harmonic(mu0, omega_trap, r, a=752e-9):
    """
    Calculate the chemical potential for a harmonic trap.

    Parameters:
    mu0 (float): Chemical potential at zero temperature.
    omega_trap (float): Trap frequency.
    r (float or array-like): Spatial position in lattice sites.
    """
    # lattice_spacing = 1064e-9 / np.sqrt(2)  # Lattice spacing in meters
    # return mu0 + 0.5 * m * (lattice_spacing * omega_trap * r ) **2

    return mu0 - 0.5 * m * (2 * np.pi * omega_trap * a * r) ** 2 / h


# Define fitting function using lmfit
def observable_model(r, mu0, T):
    return ndet_func(mu_harmonic(mu0, omega_trap, r), U, T, tx, ty)


def observable_model_all_params(r, mu0, omega_trap, U, T, tx, ty, func="ndet"):
    if func == "ndet":
        return ndet_func(mu_harmonic(mu0, omega_trap, r), U, T, tx, ty)
    if func == "n":
        return n_func(mu_harmonic(mu0, omega_trap, r), U, T, tx, ty)
    if func == "S" or func == "s":
        return s_func(mu_harmonic(mu0, omega_trap, r), U, T, tx, ty)
    else:
        raise ValueError(f"Unkown observable {func}")


def integrate_2D(rs, f_r):
    """
    Integrate a radial function f(r) over a circular area in 2D.
    
    Parameters
    ----------
    rs : array_like
        Array of radial positions (r ≥ 0)
    f_r : array_like
        Function values f(r) at each r
    
    Returns
    -------
    float
        2D integral over the circle: ∫∫ f(x,y) dx dy
    """
    integrand = f_r * rs  # multiply by r for polar coordinates
    integral = 2 * np.pi * np.trapz(integrand, rs)
    return integral


def mask_to_intervals(x, mask):
    intervals = []
    # Enumerate mask, group by True/False segments
    for key, group in groupby(enumerate(mask), key=lambda t: t[1]):
        if key:  # key=True means the mask is True in this segment
            group = list(group)
            start_index = group[0][0]
            end_index   = group[-1][0]
            intervals.append((x[start_index], x[end_index]))
    return intervals

# Example usage and fitting placeholder:
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from lmfit import Model, Parameters

    # Define synthetic or experimental density profile n(r) - 2*p_d(r)
    rs = np.linspace(0, 5, 100)  # Example spatial positions

    # True parameters (for synthetic data)
    U = 1.0  # I use it to set the scale
    mu0_true = 1.0
    T_true = U * 0.14
    omega_trap = 0.2  # ??
    tx = U / 8 / 3.8
    ty = tx

    # Generate synthetic data
    observables = observable_model_all_params(mu_harmonic(mu0_true, omega_trap, rs), U, T_true, tx, ty)

    model = Model(observable_model_all_params)
    params = Parameters()
    params.add("mu0", value=1.0)
    params.add("T", value=300.0)

    result = model.fit(observables, params, r=rs)

    # Display results
    print(result.fit_report())

    # Plot
    plt.plot(rs, observables, label="Original [n - 2p_d](r)")
    plt.plot(rs, result.best_fit, "--", label="Fitted [n - 2p_d](r)")
    plt.xlabel("r")
    plt.ylabel("n(r) - 2p_d(r)")
    plt.legend()
    plt.show()
