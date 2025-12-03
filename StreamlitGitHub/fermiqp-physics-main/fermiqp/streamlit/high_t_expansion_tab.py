"""
High-T Expansion Tab for Streamlit App
Ported from fermiqp/example.py interactive widget
"""

import numpy as np
import plotly.graph_objects as go

import fermiqp.high_t_expansion as h
import streamlit as st


def render_tab():
    """Render the high-T expansion interactive tab."""
    
    st.header("High-T expansion of the 2D Fermi-Hubbard model")
    
    def compute_curve(rs, U_Hz, mu0_Hz, omega_trap_Hz, U_over_t, T_over_t, func="n"):
        """Compute observable curve."""
        t = U_Hz / U_over_t
        T = t * T_over_t
        y = h.observable_model_all_params(rs, mu0_Hz, omega_trap_Hz, U_Hz, T, t, t, func=func)
        return y, t, T
    
    # Default values (matching example.py)
    U0_kHz = 6.0
    mu0_0_kHz = 6.0
    omega_trap_0_kHz = 0.5
    U_over_t0 = 30.0
    T_over_t0 = 0.15 * 30.0
    r_max0 = 20.0
    
    labels = {
        'n': 'n',
        's': 's / k_B',
        'ndet': 'parity projected density',
    }
    
    # Create slider controls in columns
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        U_kHz = st.slider(r"$U$ [kHz]", 0.0, 20.0, U0_kHz, step=0.1)
        mu0_kHz = st.slider(r"$\mu_0$ [kHz]", 0.0, 100.0, mu0_0_kHz, step=0.1)
        omega_kHz = st.slider(r"$\omega_{\mathrm{trap}}$ [kHz]", 0.0, 5.0, omega_trap_0_kHz, step=0.01)
    
    with col2:
        U_over_t = st.slider(r"$U/t$", 0.0, 100.0, U_over_t0, step=1.0)
        T_over_t = st.slider(r"$T/t$", 0.01, 10.0, T_over_t0, step=0.01)
        r_max = st.slider(r"$r_{\mathrm{max}}$ [sites]", 1.0, 50.0, r_max0, step=0.1)
    
    # Compute observables
    rs = np.linspace(0, r_max, min(500, int(r_max * 10)))
    U_Hz = U_kHz * 1e3
    mu0_Hz = mu0_kHz * 1e3
    omega_Hz = omega_kHz * 1e3
    
    # Compute all three observables
    try:
        y_n, t, T = compute_curve(rs, U_Hz, mu0_Hz, omega_Hz, U_over_t, T_over_t, "n")
        y_ndet, _, _ = compute_curve(rs, U_Hz, mu0_Hz, omega_Hz, U_over_t, T_over_t, "ndet")
        y_s, _, _ = compute_curve(rs, U_Hz, mu0_Hz, omega_Hz, U_over_t, T_over_t, "s")
        
        # Compute integrated quantities
        N_n = h.integrate_2D(rs, y_n)
        N_ndet = h.integrate_2D(rs, y_ndet)
        S_total = h.integrate_2D(rs, y_s)
        
    except Exception as e:
        st.error(f"Error computing observables: {e}")
        y_n = np.zeros_like(rs)
        y_ndet = np.zeros_like(rs)
        y_s = np.zeros_like(rs)
        N_n = N_ndet = S_total = 0
        t = T = 0
    
    # Display T and t table in col3
    with col3:
        import pandas as pd
        param_table = pd.DataFrame({
            "Parameter": ["t", "T"],
            "kHz": [f"{t*1e-3:.3f}", f"{T*1e-3:.3f}"]
        })
        st.dataframe(param_table, hide_index=True, width='stretch')
    
    # Create two side-by-side plots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Density (N={N_n:.0f}, N_det={N_ndet:.0f})",
            f"Entropy (Total entropy={S_total:.1f} k_B, per particle={S_total/N_n:.2f} k_B)"
        ),
        horizontal_spacing=0.12
    )
    
    # Identify MI (n≈1) and BI (n≈2) regions
    tol = 0.1
    mi_mask = np.abs(y_n - 1.0) < tol
    bi_mask = np.abs(y_n - 2.0) < tol
    
    # Add shaded regions for MI and BI
    if np.any(mi_mask):
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([rs[mi_mask], rs[mi_mask][::-1]]),
                y=np.concatenate([np.zeros_like(y_n[mi_mask]), y_n[mi_mask][::-1]]),
                fill='toself',
                fillcolor='rgba(255, 200, 0, 0.2)',
                line=dict(width=0),
                name='MI (n≈1)',
                showlegend=True,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
    
    if np.any(bi_mask):
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([rs[bi_mask], rs[bi_mask][::-1]]),
                y=np.concatenate([np.zeros_like(y_n[bi_mask]), y_n[bi_mask][::-1]]),
                fill='toself',
                fillcolor='rgba(0, 200, 255, 0.2)',
                line=dict(width=0),
                name='BI (n≈2)',
                showlegend=True,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
    
    # Left plot: Both densities
    fig.add_trace(
        go.Scatter(x=rs, y=y_n, mode='lines', name='Full density',
                   line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=rs, y=y_ndet, mode='lines', name='Parity projected',
                   line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    
    # Right plot: Entropy
    fig.add_trace(
        go.Scatter(x=rs, y=y_s, mode='lines', name='Entropy',
                   line=dict(color='green'), showlegend=False),
        row=1, col=2
    )
    
    # Update layout
    fig.update_xaxes(
        title_text="r [sites]", 
        title_font=dict(size=18, color='black'),
        showgrid=True, 
        gridcolor='rgba(128, 128, 128, 0.2)',
        zeroline=False, 
        showline=True, 
        linewidth=1, 
        linecolor='black',
        mirror=False,
        ticks='outside',
        tickfont=dict(size=12, color='black'),
        row=1, col=1
    )
    fig.update_xaxes(
        title_text="r [sites]",
        title_font=dict(size=18, color='black'),
        showgrid=True,
        gridcolor='rgba(128, 128, 128, 0.2)',
        zeroline=False,
        showline=True, 
        linewidth=1, 
        linecolor='black',
        mirror=False,
        ticks='outside',
        tickfont=dict(size=12, color='black'),
        row=1, col=2
    )
    fig.update_yaxes(
        title_text="n",
        title_font=dict(size=18, color='black'),
        showgrid=True,
        gridcolor='rgba(128, 128, 128, 0.2)',
        zeroline=False,
        showline=True, 
        linewidth=1, 
        linecolor='black',
        mirror=False,
        ticks='outside',
        range=[0, y_n.max() + 0.1],
        tickfont=dict(size=12, color='black'),
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="s / k_B",
        title_font=dict(size=18, color='black'),
        showgrid=True,
        gridcolor='rgba(128, 128, 128, 0.2)',
        zeroline=False,
        showline=True, 
        linewidth=1, 
        linecolor='black',
        mirror=False,
        ticks='outside',
        range=[0, y_s.max() + 0.1],
        tickfont=dict(size=12, color='black'),
        row=1, col=2
    )
    
    fig.update_layout(
        height=450,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(x=0.44, y=0.98, xanchor='right', yanchor='top'),
        font=dict(size=18, color='black')
    )
    
    # Update subplot titles to be black and slightly larger
    for annotation in fig.layout.annotations:
        annotation.font.size = 18
        annotation.font.color = 'black'
    
    st.plotly_chart(fig, width='stretch')
