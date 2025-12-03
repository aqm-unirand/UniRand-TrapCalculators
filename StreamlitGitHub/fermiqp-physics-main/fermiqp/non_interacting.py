from math import pi, sqrt

import numpy as np
import plotly.graph_objects as go
from scipy import integrate, interpolate, optimize, special

# -----------------------
# 1. PHYSICAL CONSTANTS
# -----------------------
hConst = 6.626e-34
mAtomic = 1.66054e-27
mLi6 = 6.015 * mAtomic

# -----------------------
# 2. HIGH-RES EQUATION OF STATE
# -----------------------
# Exact DOS (Mathematica used: (2/Pi^2)*EllipticK[1 - (e/4)^2], zero outside |e|<4)
# SciPy's ellipk expects the parameter m (same convention used here).


def rho_exact(e):
    """Density-of-states function rho(e).
    Returns 0 outside [-4,4]. For |e|<4 returns (2/pi^2)*K(1-(e/4)^2).
    The integrand has a weak (log) singularity at e=0; numerics cope by
    treating values very close to m=1 carefully.
    """
    e = float(e)
    if abs(e) >= 4.0:
        return 0.0
    m = 1.0 - (e / 4.0) ** 2
    # avoid exactly m==1 to prevent strict singularity in ellipk
    if m >= 1.0:
        m = 1.0 - 1e-12
    return (2.0 / (pi**2)) * special.ellipk(m)


# Build a super-dense grid for the integrated DOS: I(mu) = \int_{-4}^{mu} rho(e) de
mu_grid = np.arange(-4.0, 4.0 + 1e-12, 0.005)


# Integrate carefully across the van Hove point at e=0 by splitting integration if needed.
def integrate_rho_to(mu):
    if mu <= -4.0:
        return 0.0

    # small helper integrand
    def integrand(x):
        return rho_exact(x)

    # if the integration interval crosses 0, split so quad can handle the weak singularity
    if mu <= 0.0:
        res, err = integrate.quad(integrand, -4.0, mu, epsabs=1e-8, epsrel=1e-8)
        return res
    else:
        res1, err1 = integrate.quad(integrand, -4.0, 0.0, epsabs=1e-8, epsrel=1e-8)
        res2, err2 = integrate.quad(integrand, 0.0, mu, epsabs=1e-8, epsrel=1e-8)
        return res1 + res2


# Construct the raw table: pairs (mu, I(mu))
raw_table_vals = [integrate_rho_to(mu) for mu in mu_grid]

# Normalize limit so final integrated value at mu=4 corresponds to 2.0 (matching Mathematica's normalization)
max_val = raw_table_vals[-1]
norm_table_vals = [v * (2.0 / max_val) for v in raw_table_vals]

# Cubic interpolation (smooth) for the normalized integrated DOS
# Use monotonic PCHIP interpolation to avoid spline overshoot spikes
interp_I = interpolate.PchipInterpolator(mu_grid, norm_table_vals)


def get_density_smooth(mu_dim):
    if mu_dim <= -4.0:
        return 0.0
    if mu_dim >= 4.0:
        return 2.0
    return float(interp_I(mu_dim))


# -----------------------
# 3. INTEGRATION SOLVER
# -----------------------
def calc_total_N(mu_dim, kappa):
    """Calculate total atom number for given muDim and trap curvature kappa.
    Corresponds to: 2*pi * int_0^{rLimit} n(mu - kappa*r^2) * r dr
    rLimit = sqrt((mu + 4)/kappa) when mu > -4, else 0
    """
    if mu_dim <= -4.0:
        return 0.0
    r_limit_sq = (mu_dim + 4.0) / kappa
    if r_limit_sq <= 0.0:
        return 0.0
    r_limit = sqrt(r_limit_sq)

    def integrand(r):
        local_mu = mu_dim - kappa * (r**2)
        return get_density_smooth(local_mu) * r

    # Integrate with reasonable accuracy/precision goals
    integral, err = integrate.quad(integrand, 0.0, r_limit, epsabs=1e-7, epsrel=1e-7, limit=200)
    return 2.0 * pi * integral


# -----------------------
# 4. VISUALIZATION (Plotly)
# -----------------------


def solve_and_plot(nAtoms=3000, hopHz=100.0, trapHz=200.0, latSpace=0.752, show_table=True):
    """Main routine: given physical parameters, solve for mu/t and produce a plotly figure.

    Returns (fig, stats_dict) where fig is a plotly.graph_objects.Figure and stats_dict contains computed numbers.
    """
    t_val = hConst * hopHz
    k_spring = mLi6 * (2.0 * pi * trapHz) ** 2
    lat_space_SI = latSpace * 1e-6
    kappa_val = (0.5 * k_spring * lat_space_SI**2) / t_val

    # Find bracket for root: we want calc_total_N(mu) = nAtoms. Start from just above -4 and increase high bound
    def f(mu):
        return calc_total_N(mu, kappa_val) - nAtoms

    mu_low = -3.999
    mu_high = max(1.0, mu_low + 1.0)
    # increase mu_high until f(mu_high) >= 0 or until we hit a large upper bound
    for _ in range(60):
        val = f(mu_high)
        if val >= 0:
            break
        mu_high *= 2.0
        if mu_high > 1e6:
            raise RuntimeError("Unable to find an upper bracket for the root; try different parameters.")

    sol = optimize.root_scalar(f, bracket=[mu_low, mu_high], method="brentq", xtol=1e-6)
    if not sol.converged:
        raise RuntimeError("Root solver did not converge")
    mu_dim_result = sol.root

    # Fermi energy units and geometry
    fermi_energy_units = mu_dim_result / 8.0
    radius_vac = sqrt((mu_dim_result + 4.0) / kappa_val) if mu_dim_result > -4.0 else 0.0
    radius_bi = sqrt((mu_dim_result - 4.0) / kappa_val) if mu_dim_result > 4.0 else 0.0
    sites_bi = pi * (radius_bi**2)
    sites_metal = (pi * (radius_vac**2)) - sites_bi
    fill_center = get_density_smooth(mu_dim_result)
    n_check = calc_total_N(mu_dim_result, kappa_val)

    plot_limit = max(1.2 * radius_vac, 15.0)
    r_vals = np.linspace(0.0, plot_limit, 1000)
    local_mus = mu_dim_result - kappa_val * (r_vals**2)
    # evaluate density vectorized using interp (get_density_smooth handles clamps)
    density_vals = np.array([get_density_smooth(m) for m in local_mus])

    # Build plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r_vals, y=density_vals, mode="lines", name="Filling n(r)", fill="tozeroy"))

    # Add vertical BI boundary line if present
    if radius_bi > 0.5:
        fig.add_vline(
            x=radius_bi,
            line=dict(color="red", dash="dash"),
            annotation_text="BI boundary",
            annotation_position="top left",
        )
    # Add vertical VAC boundary
    if radius_vac > 0:
        fig.add_vline(
            x=radius_vac,
            line=dict(color="black", dash="dash"),
            annotation_text="vacuum boundary",
            annotation_position="top right",
        )

    # Add horizontal gridlines at n=1 and n=2
    fig.update_layout(
        title=f"6Li radial filling (N={nAtoms}, t/h={hopHz} Hz, trap={trapHz} Hz)",
        xaxis_title="Radius (Lattice Sites)",
        yaxis_title="Filling Factor (n)",
        yaxis=dict(range=[0.0, 2.1], dtick=0.5),
        template="plotly_white",
        width=800,
        height=500,
    )

    # Add annotations similar to Mathematica's Epilog
    annotations = []
    if radius_bi > 0.5:
        annotations.append(
            dict(x=radius_bi / 2, y=2.05, text="BI Core", showarrow=False, font=dict(color="red", size=12))
        )
    if radius_vac > radius_bi + 1.0:
        annotations.append(
            dict(
                x=(radius_bi + radius_vac) / 2,
                y=0.5,
                text="Metal Shell",
                showarrow=False,
                font=dict(color="blue", size=12),
            )
        )
    fig.update_layout(annotations=annotations)

    # Stats dictionary
    stats = dict(
        mu_dim=mu_dim_result,
        fermi_energy_units=fermi_energy_units,
        radius_vac=radius_vac,
        radius_bi=radius_bi,
        sites_bi=sites_bi,
        sites_metal=sites_metal,
        fill_center=fill_center,
        n_check=n_check,
        n_atoms_input=nAtoms,
        kappa=kappa_val,
    )

    # Optionally show a simple table below the plot (plotly table)
    if show_table:
        header = dict(values=["Quantity", "Value"], fill_color="lightgrey", align="left")
        rows = [
            [
                "Muon dim (mu/t)",
                "Fermi (E_F/8t)",
                "Sites BI core",
                "Sites metal shell",
                "Input N",
                "Integrated N",
                "Discrepancy",
            ],
            [
                f"{stats['mu_dim']:.6f}",
                f"{stats['fermi_energy_units']:.6f}",
                f"{stats['sites_bi']:.0f}",
                f"{stats['sites_metal']:.0f}",
                f"{stats['n_atoms_input']:.1f}",
                f"{stats['n_check']:.4f}",
                f"{abs(stats['n_atoms_input']-stats['n_check']):.4f}",
            ],
        ]
        table = go.Figure(data=[go.Table(header=header, cells=dict(values=rows))])
        # Combine as subplots by concatenating images: simpler to return both figures
        return fig, table, stats

    return fig, None, stats

def solve_for_mu(nAtoms, hopHz, trapHz, latSpace=0.75):
    """Solve for chemical potential given parameters.

    Returns:
        dict with mu_dim, fermi_energy_units, radius_vac, radius_bi,
        sites_bi, sites_metal, kappa
    """
    t_val = hConst * hopHz
    k_spring = mLi6 * (2.0 * pi * trapHz) ** 2
    lat_space_SI = latSpace * 1e-6
    kappa_val = (0.5 * k_spring * lat_space_SI**2) / t_val

    def f(mu):
        return calc_total_N(mu, kappa_val) - nAtoms

    mu_low = -3.999
    mu_high = max(1.0, mu_low + 1.0)

    # Find upper bracket
    for _ in range(60):
        val = f(mu_high)
        if val >= 0:
            break
        mu_high *= 2.0
        if mu_high > 1e6:
            raise RuntimeError("Unable to find an upper bracket for the root")

    sol = optimize.root_scalar(f, bracket=[mu_low, mu_high], method="brentq", xtol=1e-6)
    if not sol.converged:
        raise RuntimeError("Root solver did not converge")

    mu_dim_result = sol.root

    # Compute geometry
    fermi_energy_units = mu_dim_result / 8.0
    radius_vac = sqrt((mu_dim_result + 4.0) / kappa_val) if mu_dim_result > -4.0 else 0.0
    radius_bi = sqrt((mu_dim_result - 4.0) / kappa_val) if mu_dim_result > 4.0 else 0.0
    sites_bi = pi * (radius_bi**2)
    sites_metal = (pi * (radius_vac**2)) - sites_bi

    return {
        "mu_dim": mu_dim_result,
        "fermi_energy_units": fermi_energy_units,
        "radius_vac": radius_vac,
        "radius_bi": radius_bi,
        "sites_bi": sites_bi,
        "sites_metal": sites_metal,
        "kappa": kappa_val,
    }

