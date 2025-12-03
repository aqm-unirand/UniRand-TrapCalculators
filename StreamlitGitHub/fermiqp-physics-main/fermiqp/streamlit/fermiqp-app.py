import sys
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fermiqp_style import set_theme
from high_t_expansion_tab import render_tab as high_t_expansion_render_tab
from non_interacting_tab import render_tab as non_interacting_render_tab
from scipy.interpolate import interp1d

import fermiqp.lattice as ol
import streamlit as st

# import opticallattice as ol
from fermiqp.lattice import *
from fermiqp.loaddata import (
    load_lattice1d,
    load_lattice2d,
    load_scatteringlengths,
    load_zeeman,
)

V_vals_1d, t_vals_1d, wint_vals_1d = load_lattice1d()

lattice1d_t = interp1d(V_vals_1d, t_vals_1d, kind="cubic")
lattice1d_wint = interp1d(V_vals_1d, wint_vals_1d, kind="cubic")

V_vals_2d, t_vals_2d, wint_vals_2d = load_lattice2d()
lattice2d_t = interp1d(V_vals_2d, t_vals_2d, kind="cubic")
lattice2d_wint = interp1d(V_vals_2d, wint_vals_2d, kind="cubic")

st.set_page_config(
    page_title="FermiQP Physics",
    layout="wide",
)


def display_latex(latex_code):
    st.latex(latex_code)


def capture_output(func):
    global captured_output
    captured_output = []
    sys.stdout = StringIO()
    try:
        func()
    finally:
        captured_output = sys.stdout.getvalue().splitlines()
        sys.stdout = sys.__stdout__


captured_output = []


def display_as_table(output, output_text):
    headings = ["Section", "Parameter", "Value"]
    data = []
    section_name = None

    for line in output:
        if line.startswith("__________________"):
            section_name = line.strip(" _")
        elif " = " in line:
            param, value = line.split(" = ", 1)  # Split only once to avoid issues with values containing '='
            data.append([section_name, param.strip(), value.strip()])

    st.table([headings] + data)


def custom_format(val):
    if val > 5:
        return "background-color: yellow; font-weight: bold; color: red;"
    else:
        return "background-color: lightblue; text-decoration: underline;"


lattice1d = load_lattice1d()


def main():
    set_theme()

    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Fermi Gas", "Sheet Trap", "Pinning Lattice", "Superlattice", "Fields", "Lasers", "Fermi Hubbard"]
    )

    with tab0:
        st.header("Bulk Fermi gas")
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            x = float(st.text_input("Total atom number", "1000"))
        with col2:
            omega_in = float(st.text_input("In-plane trap freq (kHz)", "0.51"))
        with col3:
            omega_out = float(st.text_input("Out-of-plane trap freq (kHz)", "10.1"))
        # with col3:

        TF3D = ol.freq2temp((6 * x / 2 * omega_in**2 * omega_out) ** (1 / 3) * 1e3) * 1e6

        TF2D = ol.freq2temp((2 * x / 2 * omega_in * omega_in) ** (1 / 2) * 1e3) * 1e6

        st.write(f"Fermi temperature: {TF3D:g} μK; " + f"Fermi temperature 2D: {TF2D:g} μK")

        eta = (omega_out / omega_in) ** (2) / 2
        st.write(f"Maximal ground-band atom number: {int(eta):d}")

        st.header("Hubbard Parameters")

        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            U = float(st.text_input("U (kHz)", "10"))
        with col2:
            omega_in = float(st.text_input("In-plane trap freq (kHz)", "0.41"))
        with col3:
            omega_out = float(st.text_input("Out-of-plane trap freq (kHz)", "10.0"))

        omega_r = omega_in * 1e3
        R = np.sqrt(2 * U * 1e3 * ol.hplanck / ol.mass / (2 * np.pi * omega_r) ** 2)

        st.write(f"Mott insulator radius: {R*1e6:g} μm; Atom number for a 752 nm lattice: {int(np.pi*(R/752e-9)**2):d}")

        # x_max = st.slider("atom number", 1e3)

        # x = np.geomspace(1, x_max, 100)
        # y = ol.freq2temp((6*x/2*sheet.trap_freq(axis="radial")[0]*
        #         sheet.trap_freq(axis="radial")[1]*
        #         sheet.trap_freq(axis="axial"))**(1/3)*1e6)

        # fig, ax = plt.subplots()
        # ax.plot(x, y, color = st.color_picker("Color", "#FF0000"))

        # ax.grid()
        # ax.set_xlabel('Atom number')
        # ax.set_ylabel('Fermi temperature (μK)')
        # ax.set_xscale('log')

        # st.pyplot(fig)

        # if "df" not in st.session_state:
        #     st.session_state.df = pd.DataFrame(np.random.randn(20, 2), columns=["x", "y"])

        # st.header("Choose a datapoint color")
        # color = st.color_picker("Color", "#FF0000")
        # st.divider()
        # st.scatter_chart(st.session_state.df, x="x", y="y", color=color)

    ### Sheet trap

    with tab1:
        st.header("Sheet Trap")

        # Create columns to place inputs side by side
        (
            col1,
            col2,
            col3,
            col4,
        ) = st.columns(4)

        # Input parameters
        with col1:
            waist_vert = float(st.text_input(r"vertical waist [μm]", "4"))
        with col2:
            waist_horz = float(st.text_input(r"horizontal waist [μm]", "140"))
        with col3:
            power = float(st.text_input(r"power at atoms [W]", "40"))

        output_text = st.empty()
        capture_output(
            lambda: ol.DipoleTrap(
                waist_horz * 1e-6,
                waist_vert * 1e-6,
                U=ol.power2freq(power, waist=waist_horz * 1e-6, waisty=waist_vert * 1e-6),
            )
        )
        output_text.text("\n".join(captured_output))  # Display the captured output

        sheet = ol.DipoleTrap(
            waist_horz * 1e-6,
            waist_vert * 1e-6,
            U=ol.power2freq(power, waist=waist_horz * 1e-6, waisty=waist_vert * 1e-6),
        )

        data = {
            "": [
                r"Trap depth",
                r"Trap frequency (vertical) ",
                r"Trap frequency (horizontal) ",
                r"Trap frequency (axial) ",
                # r"$T_f$: ",
            ],
            "Energy [kHz]": [
                f"{sheet.U*1e-3:.3f}",
                f"{sheet.trap_freq(axis='radial')[1]*1e-3:2g}",
                f"{sheet.trap_freq(axis='radial')[0]*1e-3:2g}",
                f"{sheet.trap_freq(axis='axial')*1e-3:2g}",
                # f"{self.fermi_temp(N)*1e6:.3f} uK",
            ],
            "Energy [μK]": [
                f"{ol.freq2temp(sheet.U)*1e6:2g}",
                f"{ol.freq2temp(sheet.trap_freq(axis='radial')[1])*1e6:2g}",
                f"{ol.freq2temp(sheet.trap_freq(axis='radial')[0])*1e6:2g}",
                f"{ol.freq2temp(sheet.trap_freq(axis='axial'))*1e6:2g}",
                # f"{self.fermi_temp(N)*1e6:.3f} uK",
            ],
            "a_ho [nm]": [
                "",
                f"{np.sqrt(ol.hbar/ol.mass/(2*np.pi*sheet.trap_freq(axis='radial')[1]))*1e9:2g}",
                f"{np.sqrt(ol.hbar/ol.mass/(2*np.pi*sheet.trap_freq(axis='radial')[0]))*1e9:2g}",
                f"{np.sqrt(ol.hbar/ol.mass/(2*np.pi*sheet.trap_freq(axis='axial')))*1e9:2g}",
            ],
        }
        data = pd.DataFrame(data)

        # Define CSS for larger font size and spacing between tables
        css = """
            <style>
                .custom-text {
                    font-size: 24px;
                }
                .custom-table {
                    margin-top: 0px;
                    margin-bottom: 20px;
                }
            </style>
        """
        st.markdown(css, unsafe_allow_html=True)

        # Custom class for text
        # st.markdown('<p class="custom-text">Combined standard physics parameters for sheet trap</p>',
        #             unsafe_allow_html=True)
        st.write("")

        max_rows = len(data)
        data = data.reindex(range(max_rows))
        result_df = data
        result_df = result_df.fillna("")

        # Custom class for tables
        st.markdown(result_df.to_html(escape=False, index=False, classes=["custom-table"]), unsafe_allow_html=True)
        st.write("")

        st.header("Top Trap")

        # Create columns to place inputs side by side
        (
            col1,
            col2,
            col3,
            col4,
        ) = st.columns(4)

        # Input parameters
        with col1:
            waist_vert = float(st.text_input(r"waist [μm]", "50"))
        # with col2:
        # waist_horz = float(st.text_input(r"horizontal waist [μm]", "140"))
        with col2:
            power = float(st.text_input(r"power at atoms [W]", "0.4"))

        waist_horz = waist_vert

        output_text = st.empty()
        capture_output(
            lambda: ol.DipoleTrap(
                waist_horz * 1e-6,
                waist_vert * 1e-6,
                U=ol.power2freq(power, waist=waist_horz * 1e-6, waisty=waist_vert * 1e-6),
            )
        )
        output_text.text("\n".join(captured_output))  # Display the captured output

        sheet = ol.DipoleTrap(
            waist_horz * 1e-6,
            waist_vert * 1e-6,
            U=ol.power2freq(power, waist=waist_horz * 1e-6, waisty=waist_vert * 1e-6),
        )

        data = {
            "": [
                r"Trap depth",
                r"Trap frequency (vertical) ",
                r"Trap frequency (horizontal) ",
                r"Trap frequency (axial) ",
                # r"$T_f$: ",
            ],
            "Energy [kHz]": [
                f"{sheet.U*1e-3:.3f}",
                f"{sheet.trap_freq(axis='radial')[1]*1e-3:2g}",
                f"{sheet.trap_freq(axis='radial')[0]*1e-3:2g}",
                f"{sheet.trap_freq(axis='axial')*1e-3:2g}",
                # f"{self.fermi_temp(N)*1e6:.3f} uK",
            ],
            "Energy [μK]": [
                f"{ol.freq2temp(sheet.U)*1e6:2g}",
                f"{ol.freq2temp(sheet.trap_freq(axis='radial')[1])*1e6:2g}",
                f"{ol.freq2temp(sheet.trap_freq(axis='radial')[0])*1e6:2g}",
                f"{ol.freq2temp(sheet.trap_freq(axis='axial'))*1e6:2g}",
                # f"{self.fermi_temp(N)*1e6:.3f} uK",
            ],
            "a_ho [nm]": [
                "",
                f"{np.sqrt(ol.hbar/ol.mass/(2*np.pi*sheet.trap_freq(axis='radial')[1]))*1e9:2g}",
                f"{np.sqrt(ol.hbar/ol.mass/(2*np.pi*sheet.trap_freq(axis='radial')[0]))*1e9:2g}",
                f"{np.sqrt(ol.hbar/ol.mass/(2*np.pi*sheet.trap_freq(axis='axial')))*1e9:2g}",
            ],
        }
        data = pd.DataFrame(data)

        # Define CSS for larger font size and spacing between tables
        css = """
            <style>
                .custom-text {
                    font-size: 24px;
                }
                .custom-table {
                    margin-top: 0px;
                    margin-bottom: 20px;
                }
            </style>
        """
        st.markdown(css, unsafe_allow_html=True)

        # Custom class for text
        # st.markdown('<p class="custom-text">Combined standard physics parameters for sheet trap</p>',
        #             unsafe_allow_html=True)
        st.write("")

        max_rows = len(data)
        data = data.reindex(range(max_rows))
        result_df = data
        result_df = result_df.fillna("")

        # Custom class for tables
        st.markdown(result_df.to_html(escape=False, index=False, classes=["custom-table"]), unsafe_allow_html=True)
        st.write("")

    with tab2:
        st.header("Pinning Lattice")

        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        t = 0.91
        config_opts = {
            "Full interfering": {"eta_boost": (t + t**3 + t**5 + t**7) ** 2, "a": 752},
            "Retro blocked 1D pancakes": {"eta_boost": (t + t**3) ** 2, "a": 752},
            "2 x 1D lattice": {"eta_boost": (t + t**7) ** 2 + (t**3 + t**5) ** 2, "a": 532},
            "Dipole trap": {"eta_boost": (t) ** 2 + (t**3) ** 2, "a": np.nan},
        }

        # Input parameters
        with col1:
            config = st.selectbox("Lattice config:", list(config_opts.keys()))
            eta_boost = config_opts[config]["eta_boost"]
            a = config_opts[config]["a"]
        with col2:
            waist_vert = float(st.text_input(r"vert waist [μm]", "60"))
        with col3:
            waist_horz = float(st.text_input(r"horz waist [μm]", "60"))
        with col4:
            power = float(st.text_input(r"power [W]", "0.1"))

        #     output_text = st.empty()

        capture_output(lambda: ol.OpticalLattice1D(a=a * 1e-9, U=eta_boost * ol.power2freq(power)))
        output_text.text("\n".join(captured_output))  # Display the captured output

        waist_horz = waist_horz * 1e-6
        waist_vert = waist_vert * 1e-6
        if config == "2 x 1D lattice":
            latt = ol.OpticalLattice1D(
                a=a * 1e-9, U=eta_boost * ol.power2freq(power, waist=waist_horz, waisty=waist_vert) / 2
            )
        else:
            latt = ol.OpticalLattice1D(
                a=a * 1e-9, U=eta_boost * ol.power2freq(power, waist=waist_horz, waisty=waist_vert)
            )
        trap = ol.DipoleTrap(
            U=eta_boost * ol.power2freq(power, waist=waist_horz, waisty=waist_vert), waist=waist_horz, waisty=waist_vert
        )

        if config != "Dipole trap":
            st.write("")
            st.write(
                f"Lattice spacing: {a:d} nm; Lattice recoil: {latt.Er*1e-3:2g} kHz; Harmonic length: {np.sqrt(ol.hbar/ol.mass/latt.trap_freq()/2/np.pi)*1e9:.2f} nm"
            )

        data = {
            "": [
                r"Trap depth",
                r"On-site frequency",
                r"Trap frequency (vertical) ",
                r"Trap frequency (horizontal) ",
                # r"$T_f$: ",
            ],
            "Energy [kHz]": [
                f"{trap.U*1e-3:.3f}",
                f"{latt.trap_freq()*1e-3:2g}",
                f"{trap.trap_freq(axis='radial')[1]*1e-3:2g}",
                f"{trap.trap_freq(axis='radial')[0]*1e-3:2g}",
                # f"{self.fermi_temp(N)*1e6:.3f} uK",
            ],
            "Energy [μK]": [
                f"{ol.freq2temp(trap.U)*1e6:2g}",
                f"{ol.freq2temp(latt.trap_freq())*1e6:2g}",
                f"{ol.freq2temp(trap.trap_freq(axis='radial')[1])*1e6:2g}",
                f"{ol.freq2temp(trap.trap_freq(axis='radial')[0])*1e6:2g}",
                # f"{self.fermi_temp(N)*1e6:.3f} uK",
            ],
            "Energy [recoil]": [
                f"{trap.U/latt.Er:.3f}",
                f"{latt.trap_freq()/latt.Er:2g}",
                f"{trap.trap_freq(axis='radial')[1]/latt.Er:2g}",
                f"{trap.trap_freq(axis='radial')[0]/latt.Er:2g}",
                # f"{trap.trap_freq(axis='axial')/latt.Er:2g}",
                # f"{self.fermi_temp(N)*1e6:.3f} uK",
            ],
            "a_ho [nm]": [
                "",
                f"{np.sqrt(ol.hbar/ol.mass/(2*np.pi*latt.trap_freq()))*1e9:2g}",
                f"{np.sqrt(ol.hbar/ol.mass/(2*np.pi*trap.trap_freq(axis='radial')[1]))*1e9:2g}",
                f"{np.sqrt(ol.hbar/ol.mass/(2*np.pi*trap.trap_freq(axis='radial')[0]))*1e9:2g}",
                # f"{trap.trap_freq(axis='axial')/latt.Er:2g}",
                # f"{self.fermi_temp(N)*1e6:.3f} uK",
            ],
        }

        V_Er = trap.U / latt.Er
        t_Er = lattice2d_t(np.abs(V_Er))

        data = pd.DataFrame(data)

        st.markdown(css, unsafe_allow_html=True)

        # Custom class for text
        # st.markdown('<p class="custom-text">Combined standard physics parameters for pinning lattice</p>',
        #             unsafe_allow_html=True)
        st.write("")

        max_rows = len(data)
        pl_data = data.reindex(range(max_rows))
        pl_result_df = pl_data
        pl_result_df = pl_result_df.fillna("")

        # Custom class for tables
        st.markdown(pl_result_df.to_html(escape=False, index=False, classes=["custom-table"]), unsafe_allow_html=True)
        st.write("")

        st.subheader("Hubbard Parameters")

        st.write("Note: the numbers are under tight binding assumption")
        cols = st.columns([1, 1, 1.5])

        with cols[0]:
            a_sc = float(st.text_input(r"Scattering length ($a_B$)", "100.0"))
        with cols[1]:
            omega_z = float(st.text_input("Out-of-plane confinement (kHz)", "30.0"))

        V0 = np.abs(trap.U / latt.Er)

        a_z = np.sqrt(ol.hbar / ol.mass / omega_z / 1000 / 2 / np.pi)
        a_x = a * 1e-9 / np.pi * (1 / V0) ** (1 / 4)

        # print([a_x, a_z])
        U = 2 * ol.hbar * a_sc * 5.29e-11 / ol.mass / (8 * np.pi**1.5 * a_x**2 * a_z)

        U = U * 4  # ???

        # U = np.sqrt(8*np.pi)*a_sc/(a)*V0**(3/4) * latt.Er

        t = 4 / np.sqrt(np.pi) * V0 ** (3 / 4) * np.exp(-2 * np.sqrt(V0)) * latt.Er

        data = {
            r"hopping t (kHz)": [t_Er * latt.Er * 1e-3],
            r"interaction U (kHz)": [U * 1e-3],
            r"superexchange (kHz)": [4 * t**2 / U * 1e-3],
        }
        data = pd.DataFrame(data)

        st.markdown(css, unsafe_allow_html=True)

        # Custom class for text
        # st.markdown('<p class="custom-text">Combined standard physics parameters for pinning lattice</p>',
        #             unsafe_allow_html=True)
        st.write("")

        max_rows = len(data)
        pl_data = data.reindex(range(max_rows))
        pl_result_df = pl_data
        pl_result_df = pl_result_df.fillna("")

        # Custom class for tables
        st.markdown(pl_result_df.to_html(escape=False, index=False, classes=["custom-table"]), unsafe_allow_html=True)
        st.write("")

    with tab3:
        st.header("Superlattice")

        st.markdown("### Green")

        cols = st.columns([1, 1, 1, 1])

        with cols[1]:
            waist_vert = float(st.text_input(r"vert waist [μm]", "113"))
        with cols[2]:
            waist_horz = float(st.text_input(r"horz waist [μm]", "233"))
        with cols[0]:
            power = float(st.text_input(r"power per beam [W]", "1"))
        with cols[3]:
            a = float(st.text_input(r"Spacing [μm]", "3.05"))
            # pass

        eta_boost = 4

        a = a * 1e3

        wavelen = 532e-9

        capture_output(
            lambda: ol.OpticalLattice1D(
                a=a * 1e-9,
                U=eta_boost * ol.power2freq(power, waist=waist_horz * 1e-6, waisty=waist_vert * 1e-6, wavelen=wavelen),
            )
        )
        output_text.text("\n".join(captured_output))  # Display the captured output

        latt = ol.OpticalLattice1D(
            a=a * 1e-9,
            U=eta_boost * ol.power2freq(power, waist=waist_horz * 1e-6, waisty=waist_vert * 1e-6, wavelen=wavelen),
        )
        trap = ol.DipoleTrap(
            U=eta_boost * ol.power2freq(power, waist=waist_horz * 1e-6, waisty=waist_vert * 1e-6, wavelen=wavelen),
            waist=waist_horz * 1e-6,
            waisty=waist_vert * 1e-6,
        )

        st.write(
            f"Lattice spacing: {int(a):d} nm; Lattice recoil: {latt.Er*1e-3:2g} kHz; Harmonic length: {np.sqrt(ol.hbar/ol.mass/latt.trap_freq()/2/np.pi)*1e9:.2f} nm"
        )

        data = {
            "": [
                r"Trap depth",
                r"On-site frequency",
                r"Trap frequency (vertical) ",
                r"Trap frequency (horizontal) ",
                # r"$T_f$: ",
            ],
            "Energy [kHz]": [
                f"{trap.U*1e-3:.3f}",
                f"{latt.trap_freq()*1e-3:2g}",
                f"{trap.trap_freq(axis='radial')[1]*1e-3:2g}",
                f"{trap.trap_freq(axis='radial')[0]*1e-3:2g}",
                # f"{self.fermi_temp(N)*1e6:.3f} uK",
            ],
            "Energy [μK]": [
                f"{ol.freq2temp(trap.U)*1e6:2g}",
                f"{ol.freq2temp(latt.trap_freq())*1e6:2g}",
                f"{ol.freq2temp(trap.trap_freq(axis='radial')[1])*1e6:2g}",
                f"{ol.freq2temp(trap.trap_freq(axis='radial')[0])*1e6:2g}",
                # f"{self.fermi_temp(N)*1e6:.3f} uK",
            ],
            "Energy [recoil]": [
                f"{trap.U/latt.Er:.3f}",
                f"{latt.trap_freq()/latt.Er:2g}",
                f"{trap.trap_freq(axis='radial')[1]/latt.Er:2g}",
                f"{trap.trap_freq(axis='radial')[0]/latt.Er:2g}",
                # f"{trap.trap_freq(axis='axial')/latt.Er:2g}",
                # f"{self.fermi_temp(N)*1e6:.3f} uK",
            ],
            "a_ho [nm]": [
                "",
                f"{np.sqrt(ol.hbar/ol.mass/(2*np.pi*latt.trap_freq()))*1e9:2g}",
                f"{np.sqrt(ol.hbar/ol.mass/(2*np.pi*trap.trap_freq(axis='radial')[1]))*1e9:2g}",
                f"{np.sqrt(ol.hbar/ol.mass/(2*np.pi*trap.trap_freq(axis='radial')[0]))*1e9:2g}",
                # f"{trap.trap_freq(axis='axial')/latt.Er:2g}",
                # f"{self.fermi_temp(N)*1e6:.3f} uK",
            ],
        }
        data = pd.DataFrame(data)

        st.markdown(css, unsafe_allow_html=True)

        # Custom class for text
        # st.markdown('<p class="custom-text">Combined standard physics parameters for pinning lattice</p>',
        #             unsafe_allow_html=True)
        st.write("")

        max_rows = len(data)
        pl_data = data.reindex(range(max_rows))
        pl_result_df = pl_data
        pl_result_df = pl_result_df.fillna("")

        # Custom class for tables
        st.markdown(pl_result_df.to_html(escape=False, index=False, classes=["custom-table"]), unsafe_allow_html=True)
        st.write("")

        st.markdown("### IR")

        cols = st.columns([1, 1, 1, 1])

        with cols[1]:
            waist_vert = float(st.text_input(r"vert waist [μm]", "130."))
        with cols[2]:
            waist_horz = float(st.text_input(r"horz waist [μm]", "300."))
        with cols[0]:
            power = float(st.text_input(r"power per beam [W]", "2."))
        with cols[3]:
            a = float(st.text_input(r"Spacing [μm]", "6.1"))
            # pass

        eta_boost = 4

        a = a * 1e3

        wavelen = 1064e-9

        capture_output(
            lambda: ol.OpticalLattice1D(
                a=a * 1e-9,
                U=eta_boost * ol.power2freq(power, waist=waist_horz * 1e-6, waisty=waist_vert * 1e-6, wavelen=wavelen),
            )
        )
        output_text.text("\n".join(captured_output))  # Display the captured output

        latt = ol.OpticalLattice1D(
            a=a * 1e-9,
            U=eta_boost * ol.power2freq(power, waist=waist_horz * 1e-6, waisty=waist_vert * 1e-6, wavelen=wavelen),
        )
        trap = ol.DipoleTrap(
            U=eta_boost * ol.power2freq(power, waist=waist_horz * 1e-6, waisty=waist_vert * 1e-6, wavelen=wavelen),
            waist=waist_horz * 1e-6,
            waisty=waist_vert * 1e-6,
        )

        st.write(
            f"Lattice spacing: {int(a):d} nm; Lattice recoil: {latt.Er*1e-3:2g} kHz; Harmonic length: {np.sqrt(ol.hbar/ol.mass/latt.trap_freq()/2/np.pi)*1e9:.2f} nm"
        )

        data = {
            "": [
                r"Trap depth",
                r"On-site frequency",
                r"Trap frequency (vertical) ",
                r"Trap frequency (horizontal) ",
                # r"$T_f$: ",
            ],
            "Energy [kHz]": [
                f"{trap.U*1e-3:.3f}",
                f"{latt.trap_freq()*1e-3:2g}",
                f"{trap.trap_freq(axis='radial')[1]*1e-3:2g}",
                f"{trap.trap_freq(axis='radial')[0]*1e-3:2g}",
                # f"{self.fermi_temp(N)*1e6:.3f} uK",
            ],
            "Energy [μK]": [
                f"{ol.freq2temp(trap.U)*1e6:2g}",
                f"{ol.freq2temp(latt.trap_freq())*1e6:2g}",
                f"{ol.freq2temp(trap.trap_freq(axis='radial')[1])*1e6:2g}",
                f"{ol.freq2temp(trap.trap_freq(axis='radial')[0])*1e6:2g}",
                # f"{self.fermi_temp(N)*1e6:.3f} uK",
            ],
            "Energy [recoil]": [
                f"{trap.U/latt.Er:.3f}",
                f"{latt.trap_freq()/latt.Er:2g}",
                f"{trap.trap_freq(axis='radial')[1]/latt.Er:2g}",
                f"{trap.trap_freq(axis='radial')[0]/latt.Er:2g}",
                # f"{trap.trap_freq(axis='axial')/latt.Er:2g}",
                # f"{self.fermi_temp(N)*1e6:.3f} uK",
            ],
            "a_ho [nm]": [
                "",
                f"{np.sqrt(ol.hbar/ol.mass/(2*np.pi*latt.trap_freq()))*1e9:2g}",
                f"{np.sqrt(ol.hbar/ol.mass/(2*np.pi*trap.trap_freq(axis='radial')[1]))*1e9:2g}",
                f"{np.sqrt(ol.hbar/ol.mass/(2*np.pi*trap.trap_freq(axis='radial')[0]))*1e9:2g}",
                # f"{trap.trap_freq(axis='axial')/latt.Er:2g}",
                # f"{self.fermi_temp(N)*1e6:.3f} uK",
            ],
        }
        data = pd.DataFrame(data)

        st.markdown(css, unsafe_allow_html=True)

        # Custom class for text
        # st.markdown('<p class="custom-text">Combined standard physics parameters for pinning lattice</p>',
        #             unsafe_allow_html=True)
        st.write("")

        max_rows = len(data)
        pl_data = data.reindex(range(max_rows))
        pl_result_df = pl_data
        pl_result_df = pl_result_df.fillna("")

        # Custom class for tables
        st.markdown(pl_result_df.to_html(escape=False, index=False, classes=["custom-table"]), unsafe_allow_html=True)
        st.write("")

    with tab4:
        st.header("Magnetic Fields")

        cols = st.columns([3, 1])

        with cols[0]:
            B0 = st.slider("Field (G)", 0, 1200, 600)

        st.subheader("Scattering lengths")

        B, a_sc = load_scatteringlengths()

        # unit = st.radio("unit", [r"$a_B$", "nm"])

        # if unit == "nm":
        #      a_sc *= 0.0529177

        cols = st.columns([3, 1])

        with cols[0]:
            fig, ax = plt.subplots(figsize=(6, 3))

            iB0 = np.argmin(np.abs(B - B0))

            labels = ["1-2", "1-3", "2-3"]
            for i in range(3):
                a_sc0 = a_sc[iB0, i]

                ax.plot(B, a_sc[:, i], color=f"C{i}")
                ax.plot(B0, a_sc0, "o", color=f"C{i}", label=f"{labels[i]}: {a_sc0: .1f} " + r"$a_B$")

            ax.legend()
            ax.grid()
            # if unit == "nm":
            #     ax.set_ylim([-50, 100])
            # else:
            #     ax.set_ylim([-1000, 2000])
            ax.set_ylim([-1000, 2000])
            ax.set_xlim([0, 1200])
            ax.set_xlabel("B field (G)")
            ax.set_ylabel("Scattering length")

            st.pyplot(fig)

        st.subheader("RF & MW transitions")

        cols = st.columns([1, 1])

        with cols[0]:
            fig, ax = plt.subplots(figsize=(4, 3))

            B, Eg = load_zeeman()
            iB0 = np.argmin(np.abs(B - B0))

            Eg = Eg / 1e6

            E12 = Eg[:, 1] - Eg[:, 0]
            E23 = Eg[:, 2] - Eg[:, 1]

            ax.plot(B, E12 - 80, "-", label=f"1-2: {E12[iB0]:.3f} MHz")
            ax.plot(B, E23 - 80, "-", label=f"2-3: {E23[iB0]:.3f} MHz")

            ax.plot(B0, E12[iB0] - 80, "o", color="C0")
            ax.plot(B0, E23[iB0] - 80, "o", color="C1")

            ax.legend()
            ax.grid()
            ax.set_ylim([-10, 10])
            ax.set_xlim([0, 1200])
            ax.set_xlabel("B field (G)")
            ax.set_ylabel("Trans. Freq. - 80 [MHz]")

            st.pyplot(fig)

        with cols[1]:
            fig, ax = plt.subplots(figsize=(4, 3))

            B, Eg = load_zeeman()
            iB0 = np.argmin(np.abs(B - B0))

            Eg = Eg / 1e9

            for i in range(3):
                E = Eg[:, -i - 1] - Eg[:, i]

                ax.plot(B, E, "-", label=f"{i+1}-{6-i}: {E[iB0]:.3f} GHz")

                ax.plot(B0, E[iB0], "o", color=f"C{i}")

            ax.legend()
            ax.grid()
            ax.set_ylim([0, 3])
            ax.set_xlim([0, 1200])
            ax.set_xlabel("B field (G)")
            ax.set_ylabel("Trans. Freq. [GHz]")

            st.pyplot(fig)

        st.subheader("Closed transitions")

        st.markdown("Ref: D2 cooler")

        cols = st.columns([1, 1])

        with cols[0]:
            fig, ax = plt.subplots(figsize=(4, 3))

            B, Ee = load_zeeman("2P")
            B, Eg = load_zeeman("2S")
            iB0 = np.argmin(np.abs(B - B0))

            Eg = Eg / 1e9
            Ee = Ee / 1e9

            ref = Ee[0, :].max() - Eg[0, :].max()

            for i in range(3):
                E = Ee[:, 6 + i] - Eg[:, i] - ref

                ax.plot(B, E, "-", label=rf"${i+1}- |-3/2, {1-i}\rangle$: {E[iB0]*1000:.0f} MHz")

                ax.plot(B0, E[iB0], "o", color=f"C{i}")

            ax.legend()
            ax.grid()
            ax.set_ylim([-2, 0])
            ax.set_xlim([0, 1200])
            ax.set_xlabel("B field (G)")
            ax.set_ylabel("Trans. Freq. [GHz]")

            st.pyplot(fig)

        with cols[1]:
            fig, ax = plt.subplots(figsize=(4, 3))

            B, Ee = load_zeeman("2P")
            B, Eg = load_zeeman("2S")
            iB0 = np.argmin(np.abs(B - B0))

            Eg = Eg / 1e9
            Ee = Ee / 1e9

            ref = Ee[0, :].max() - Eg[0, :].max()

            for i in range(3):
                E = Ee[:, -3 + i] - Eg[:, -3 + i] - ref

                ax.plot(B, E, "-", label=rf"${i+4}- |3/2, {1-i}\rangle$: {E[iB0]*1000:.0f} MHz")

                ax.plot(B0, E[iB0], "o", color=f"C{i}")

            ax.legend()
            ax.grid()
            ax.set_ylim([0, 2])
            ax.set_xlim([0, 1200])
            ax.set_xlabel("B field (G)")
            ax.set_ylabel("Trans. Freq. [GHz]")

            st.pyplot(fig)
        # st.subheader("Field mapings")

        # cols = st.columns([1,1,4])

        # with cols[0]:
        #     delta_I = float(st.text_input(r"$\Delta I$ (A)", "0"))
        # grad_B = delta_I * 0.454/2
        # # grad_E = grad_B * 1.4

        # with cols[1]:
        #     distance = float(st.text_input(r"$d$ (mum)", "20"))

        # with cols[2]:
        #     st.markdown(r"$\partial_y B$")
        #     # st.markdown(f"{grad_B:g} G/cm; {grad_B*1e-4*1.36e3:g} kHz/mum")
        #     st.markdown(r"$\Delta B$" + f": {distance*grad_B*1e-4} G; " + r"$\Delta E$" + f": {distance*grad_B*1e-4*1.36e3} kHz")

    #     # Create columns to place inputs side by side

    #     col1, col2, col3, col4, = st.columns(4)

    #     # Input parameters
    #     with col1:
    #         pl_waist = float(st.text_input(r"Lattice horizontal waist [$\mu m$]", "60"))
    #     with col2:
    #         pl_power = float(st.text_input(r"Lattice input beam power [$W$]", "80"))
    #     with col3:
    #         pl_waist_vert= float(st.text_input(r"Lattice vertical waist [$\mu m$]", "60"))
    #     with col4:
    #         pl_N_atoms = float(st.text_input(r"Lattice Number of atoms ", "200000"))

    #     output_text = st.empty()
    #     capture_output(lambda: OpticalLattice(Atom(name="li6", mF=-1/2,F="lower"),
    #                                             P=pl_power,
    #                                             w0=pl_waist*1e-6,
    #                                             lamLas=1064e-9,
    #                                             polLas=0,
    #                                             dLattice=752e-9,
    #                                             name="Pinning Lattice",
    #                                             V0_er="get",).PrintResults())

    #     output_text.text("\n".join(captured_output))  # Display the captured output

    #     data,d,d,    = OpticalLattice(Atom(name="li6",mF=-1/2, F="lower"),
    #                                     P=pl_power,
    #                                     w0=pl_waist_hor*1e-6,
    #                                     lamLas=1064e-9,
    #                                     polLas=0,
    #                                     dLattice=752e-9,
    #                                     name="Pinning Lattice",
    #                                     V0_er="get",).PrintResults()
    #     st.markdown(css, unsafe_allow_html=True)

    #     # Custom class for text
    #     st.markdown('<p class="custom-text">Combined standard physics parameters for pinnin lattice</p>',
    #                 unsafe_allow_html=True)
    #     st.write('')

    #     max_rows = len(data)
    #     data = data.reindex(range(max_rows))
    #     pl_result_df = data
    #     pl_result_df = pl_result_df.fillna('')

    #     # Custom class for tables
    #     st.markdown(pl_result_df.to_html(escape=False, index=False, classes=['custom-table']), unsafe_allow_html=True)
    #     st.write('')

    # with tab4:
    #     st.header('Saturation Int. D1 Line')

    #     # Create columns to place inputs side by side

    #     col1, col2, = st.columns(2)

    #     # Input parameters
    #     with col1:
    #         d1_waist = float(st.text_input(r"D1 horizontal waist [$\mu m$]", "500"))
    #     with col2:
    #         d1_power = float(st.text_input(r"D1 beam power [$mW$]", "1"))

    #     output_text = st.empty()

    #     def calc_I(P,waist):
    #         return 2*P*1e-3/(OpticalTrap.pi*(waist*1e-6)**2)
    #     def calc_I_sat(P,waist,Isat):
    #         '''Waist in um and P in mW'''
    #         int_d1 = calc_I(P,waist)
    #         return int_d1,int_d1/Isat

    #     capture_output(lambda: calc_I_sat(d1_power,d1_waist,OpticalTrap.li6.IsatD1))

    #     output_text.text("\n".join(captured_output))  # Display the captured output

    #     d1_result = calc_I_sat(d1_power,d1_waist,OpticalTrap.li6.IsatD1)

    #     st.markdown(css, unsafe_allow_html=True)

    #     # Custom class for tables
    #     st.markdown(f"Using $I_{{sat}} = {OpticalTrap.li6.IsatD1/1e4*1e3:1.2f}mW/cm^2$")
    #     st.markdown(f"$I = $ {d1_result[0]/1e4*1e3:1.1f} $mW/cm^2$")
    #     st.markdown(f"$I/I_{{sat}} = $ {d1_result[1]:1.1f}")

    #     st.write('')

    #     st.header('Saturation Int. D2 Line')

    #     # Create columns to place inputs side by side

    #     col21, col22, = st.columns(2)

    #     # Input parameters
    #     with col21:
    #         d2_waist = float(st.text_input(r"D2 horizontal waist [$\mu m$]", "500"))
    #     with col22:
    #         d2_power = float(st.text_input(r"D2 beam power [$mW$]", "1"))

    #     output_text = st.empty()

    #     capture_output(lambda: calc_I_sat(d2_power,d2_waist,OpticalTrap.li6.IsatD2))

    #     output_text.text("\n".join(captured_output))  # Display the captured output

    #     d1_result = calc_I_sat(d2_power,d2_waist,OpticalTrap.li6.IsatD2)

    #     st.markdown(css, unsafe_allow_html=True)

    #     # Custom class for tables
    #     st.markdown(f"Using $I_{{sat}} = {OpticalTrap.li6.IsatD2/1e4*1e3:1.2f}mW/cm^2$")
    #     st.markdown(f"$I = $ {d1_result[0]/1e4*1e3:1.1f} $mW/cm^2$")
    #     st.markdown(f"$I/I_{{sat}} = $ {d1_result[1]:1.1f}")

    #     st.write('')

    with tab5:
        st.subheader("Intensities")

        cols = st.columns([1, 1, 1, 4])

        with cols[0]:
            P = float(st.text_input(r"Power [mW]", "1"))

        with cols[1]:
            w = float(st.text_input(r"Beam radius (waist) [mm]", "1"))

        with cols[2]:
            config_opts = {
                "D2 Line": {"Isat": 2.54},
                "D1 Line": {"Isat": 5.72},
            }
            config = st.selectbox("Line:", list(config_opts.keys()))
            Isat = config_opts[config]["Isat"]

        I = 2 * P / np.pi / w**2 * 100
        s0 = I / Isat
        Omega = np.sqrt(s0 / 2) * 6

        st.write(rf"$s_0$ = {s0:g}; $\Omega/2\pi$ = {Omega:g} MHz")

        fig, ax = plt.subplots(figsize=(3, 2))

        x = np.linspace(0, 10, 100)
        y = np.geomspace(0.01, 100, 100)

        X, Y = np.meshgrid(x, y)

        v = 0.5 * s0 * Y / (1 + s0 * Y + 4 * X**2)

        # ax.plot(x,
        im = ax.contourf(
            X, Y, v, levels=[0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], cmap="diverging"
        )
        plt.colorbar(im, label=r"$R_{\rm sc}/\Gamma$")

        ax.set_xlabel(r"$\Delta/\Gamma$")
        ax.set_ylabel(r"$s/s_0$")
        ax.set_yscale("log")
        # ax.set_yticks([0.05, 0.1, 0.5])

        st.pyplot(fig, width="stretch")

        st.subheader("Detunings")

        cols = st.columns([1, 1, 1, 4])

        with cols[0]:
            f_cooler = float(st.text_input(r"Cooler AOM [MHz]", "85.5"))

        with cols[1]:
            f_repumper = float(st.text_input(r"Repumper AOM [MHz]", "199.6"))

        with cols[2]:
            f_d1 = float(st.text_input(r"Socrates offset lock [MHz]", "14.6"))

        data = {
            "": [
                r"Detunings [MHz]",
                # r"$T_f$: ",
            ],
            "D2 Cooler [MHz]": [
                f"{(f_cooler-85.5)*2:.1f}",
            ],
            "D2 Repumper [MHz]": [
                f"{(f_repumper-199.6)*2:.1f}",
            ],
            "D1 Cooler [MHz]": [
                f"{(f_cooler-85.5)*2-(f_d1-14.6)*10:.1f}",
            ],
            "D1 Repumper [MHz]": [
                f"{(f_repumper-199.6)*2-(f_d1-14.6)*10:.1f}",
            ],
        }
        data = pd.DataFrame(data)

        st.markdown(css, unsafe_allow_html=True)
        # Custom class for text
        # st.markdown('<p class="custom-text">Combined standard physics parameters for pinning lattice</p>',
        #             unsafe_allow_html=True)
        st.write("")

        max_rows = len(data)
        pl_data = data.reindex(range(max_rows))
        pl_result_df = pl_data
        pl_result_df = pl_result_df.fillna("")

        # Custom class for tables
        st.markdown(pl_result_df.to_html(escape=False, index=False, classes=["custom-table"]), unsafe_allow_html=True)
        st.write("")

    with tab6:
        high_t_expansion_render_tab()

        st.markdown("---")  # Separator

        non_interacting_render_tab()


if __name__ == "__main__":
    main()
