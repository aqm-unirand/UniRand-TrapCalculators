"""
Non-Interacting Fermi-Hubbard Model Tab for Streamlit App
Solves for density profiles in trapped non-interacting Fermi gas
"""

from math import pi, sqrt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import optimize

import streamlit as st
from fermiqp.non_interacting import (
    get_density_smooth,
    solve_for_mu,
)


def render_tab():
    """Render the non-interacting Fermi-Hubbard model tab."""

    st.header("Non-interacting Fermi-Hubbard model")
    st.markdown("Radial density profile for trapped non-interacting fermions in optical lattice")

    # Default values
    nAtoms0 = 3000
    hopHz0 = 100.0
    trapHz0 = 200.0
    latSpace = 0.752  # micrometers, fixed

    # Create slider controls
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        nAtoms = st.slider("Atom Number", 100, 10000, nAtoms0, step=100)
        hopHz = st.slider("Hopping t/h [Hz]", 10.0, 500.0, hopHz0, step=10.0)
        trapHz = st.slider("Trap Frequency [Hz]", 50.0, 500.0, trapHz0, step=10.0)

    # Solve for mu and geometry
    try:
        stats = solve_for_mu(nAtoms, hopHz, trapHz, latSpace)

        # Display results table in col2
        with col2:
            results_table = pd.DataFrame(
                {
                    "Quantity": ["Fermi Energy (E_F/8t)", "Sites in BI core", "Sites in metal shell"],
                    "Value": [
                        f"{stats['fermi_energy_units']:.4f}",
                        f"{stats['sites_bi']:.0f}",
                        f"{stats['sites_metal']:.0f}",
                    ],
                }
            )
            st.dataframe(results_table, hide_index=True, width="stretch")

        # Compute density profile
        mu_dim = stats["mu_dim"]
        kappa = stats["kappa"]
        radius_vac = stats["radius_vac"]
        radius_bi = stats["radius_bi"]

        plot_limit = max(1.2 * radius_vac, 15.0)
        r_vals = np.linspace(0.0, plot_limit, 1000)
        local_mus = mu_dim - kappa * (r_vals**2)
        density_vals = np.array([get_density_smooth(m) for m in local_mus])

        # Create plotly figure
        fig = go.Figure()

        # Add shaded regions for BI and metal phases
        # BI phase: r < radius_bi
        if radius_bi > 0.5:
            r_bi = r_vals[r_vals <= radius_bi]
            density_bi = density_vals[: len(r_bi)]
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([r_bi, r_bi[::-1]]),
                    y=np.concatenate([np.zeros_like(density_bi), density_bi[::-1]]),
                    fill="toself",
                    fillcolor="rgba(0, 200, 255, 0.2)",
                    line=dict(width=0),
                    name="BI phase",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

        # Metal phase: radius_bi < r < radius_vac
        if radius_vac > radius_bi + 0.5:
            r_metal = r_vals[(r_vals > radius_bi) & (r_vals <= radius_vac)]
            density_metal = density_vals[len(r_vals[r_vals <= radius_bi]) : len(r_vals[r_vals <= radius_vac])]
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([r_metal, r_metal[::-1]]),
                    y=np.concatenate([np.zeros_like(density_metal), density_metal[::-1]]),
                    fill="toself",
                    fillcolor="rgba(100, 255, 100, 0.2)",
                    line=dict(width=0),
                    name="Metal phase",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

        # Main density trace
        fig.add_trace(
            go.Scatter(
                x=r_vals,
                y=density_vals,
                mode="lines",
                name="Filling n(r)",
                line=dict(color="blue", width=2),
                showlegend=False,
            )
        )

        # Add vertical lines for boundaries
        if radius_bi > 0.5:
            fig.add_vline(
                x=radius_bi,
                line=dict(color="red", dash="dash", width=2),
                annotation_text="BI boundary",
                annotation_position="top left",
                annotation=dict(font_size=18, font_color="red"),
            )

        if radius_vac > 0:
            fig.add_vline(
                x=radius_vac,
                line=dict(color="black", dash="dash", width=2),
                annotation_text="vacuum boundary",
                annotation_position="top right",
                annotation=dict(font_size=18, font_color="black"),
            )

        # Update layout with consistent styling
        fig.update_xaxes(
            title_text="r [sites]",
            title_font=dict(size=18, color="black"),
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.2)",
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=False,
            ticks="outside",
            tickfont=dict(size=12, color="black"),
        )

        fig.update_yaxes(
            title_text="n",
            title_font=dict(size=18, color="black"),
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.2)",
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=False,
            ticks="outside",
            range=[0, 2.1],
            dtick=0.5,
            tickfont=dict(size=12, color="black"),
        )

        fig.update_layout(
            height=450,
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=True,
            legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top"),
            font=dict(size=18, color="black"),
        )

        # Add region label for BI core only
        annotations = []
        if radius_bi > 0.5:
            annotations.append(
                dict(
                    x=radius_bi / 2,
                    y=2.05,
                    text="BI Core",
                    showarrow=False,
                    font=dict(color="blue", size=14, family="Arial Black"),
                )
            )

        fig.update_layout(annotations=annotations)

        st.plotly_chart(fig, width="stretch")

    except Exception as e:
        st.error(f"Error computing solution: {e}")
