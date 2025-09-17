import pandas as pd
import streamlit as st
import numpy as np
import math
import scipy.constants as const
from arc import *


def joule_to_hz(energy_in_joule):
    return energy_in_joule / (2 * const.pi * const.hbar)

def hz_to_joule(energy_in_hz):
    return energy_in_hz * (2 * const.pi * const.hbar)

def wavelength_to_freq(wavelength_in_m):
    return (const.c)/wavelength_in_m 

# Constants
m_li6 = 6.0151228874 * 1.66e-27  # Mass of Lithium-6 in kg
transition_linewidth = 2 * const.pi*5.8724e6 #Li-6 D2 transition linewidth
transition_wavelength = 670.977338e-9 #Li-6 transition wavelength in m
transition_frequency=2 * const.pi*wavelength_to_freq(transition_wavelength)

# Streamlit app
st.title('UniRand Cross ODT Parameters for Li6')

# Sidebar input fields
st.sidebar.header('Input Parameters')

power_slider = st.sidebar.slider('IPG Output Power, mW', min_value=0.1, max_value=5000.0, value=1000.0, step=0.1)
waist_slider = st.sidebar.slider('Beam Waist, μm', min_value=1.0, max_value=200.0, value=67.0, step=0.5)

# Wavelength
trap_wavelength_input = st.sidebar.number_input('Wavelength, nm', value=1070)
trap_wavelength_input *= 1e-9  # Convert to meters
trap_frequency=2 * const.pi*wavelength_to_freq(trap_wavelength_input)

# Power
odt_beam_power_input = st.sidebar.number_input('Laser Power, mW', value=power_slider)
odt_beam_power_input *= 1e-3 #convert power to W

# Waist
odt_beam_waist_input = st.sidebar.number_input('Beam Waist, μm', value=waist_slider)
odt_beam_waist_input *= 1e-6  # Convert to meters

# Cross angle
cross_odt_angle_input = st.sidebar.number_input('Cross angle, degrees', value=10)

# Atom Number
atom_number_input = st.sidebar.number_input('Atom Number, N', value=4000)


# Cache the polarizability calculation to optimize performance
@st.cache_data
def getApproxPolarizability(traplight_frequency,atom_transition_freq,atom_transition_linewidth):
    num=6*const.pi*const.epsilon_0*(const.c**3)*(atom_transition_linewidth/(atom_transition_freq**2))
    denRe=(atom_transition_freq**2)-(traplight_frequency**2)
    denIm=1j*(traplight_frequency**3/atom_transition_freq**2)*atom_transition_linewidth
    alpha=num/(denRe-denIm)
    return alpha

    
# Calculate U_0 (Trap depth)
def get_U0(trap_freq,power, waist):
    alpha=getApproxPolarizability(trap_freq,transition_frequency,transition_linewidth)
    total_power_in_cross = 2*power #Total power in the cross is twice the IPG output power
    trapIntensity=2*total_power_in_cross/(const.pi*(waist**2)) #Max intensity on optical axis is 2 times power/mode area for Gaussian beam
    Udip=alpha.real*trapIntensity #((-1/(2*const.epsilon_0*const.c))*alpha.real*trapIntensity)
    return Udip*1e30 #Udip #Depth in J

# Calculate Cross ODT trap frequencies (omega_x, omega_y, omega_z)
def get_cross_freqs(trap_freq,power,waist,cross_angle):
    RayleighRange = (const.pi*(waist**2))/(trap_wavelength_input)
    Uin = get_U0(trap_freq,power, waist)* 0.5
    Uout = Uin#self.getTrapDepth()* 0.5
    wr = np.sqrt(4*np.abs(Uin+Uout)/(m_li6 * waist * waist))
    wz = (np.sqrt(2*np.abs(Uin+Uout)/(m_li6 * RayleighRange* RayleighRange)))
    wv = (wr*np.sqrt(2))/(2 * const.pi)
    wx = (np.sqrt((2*((wz*np.cos(np.deg2rad(cross_angle/2)))**2)) + (2*((wr*np.sin(np.deg2rad(cross_angle/2)))**2))))/(2 * const.pi)
    wy = (np.sqrt((2*((wr*np.cos(np.deg2rad(cross_angle/2)))**2)) + (2*((wz*np.sin(np.deg2rad(cross_angle/2)))**2))))/(2 * const.pi)
    return wx,wy,wv #Output values in Hz

# Calculate Fermi Characteristics of the trap
def get_Fermi_values(trap_freq,power,waist,cross_angle, atom_number):
    wx,wy,wz = get_cross_freqs(trap_freq,power,waist,cross_angle)
    Ef = (const.h*((wx*wy*wz)**(1/3))*((3*atom_number)**(1/3)))
    Tf = Ef/const.k
    vf = np.sqrt(2*Ef/m_li6)
    return Ef,Tf,vf #Energy in J, Temperature in K, velocity in m/s 


# Calculate and display results
U0 = get_U0(trap_wavelength_input, odt_beam_power_input, odt_beam_waist_input) # U0 in joules

# Radial (wx,wy) and axial trap (wv) frequencies
wx,wy,wv = get_cross_freqs(trap_wavelength_input, odt_beam_power_input, odt_beam_waist_input,cross_odt_angle_input)

#Calculate Fermi Characteristics Ef (fermi Energy), Tf (Fermi Temperature), vf (Fermi velocity)
Ef,Tf,vf = get_Fermi_values(trap_wavelength_input, odt_beam_power_input, odt_beam_waist_input,cross_odt_angle_input,atom_number_input)

alpha = getApproxPolarizability(trap_frequency,transition_frequency,transition_linewidth)*1e39

on = st.sidebar.toggle("Switch to temperature")

# Create a dataframe for displaying results
# if not on:
#     data = pd.DataFrame({
#         "Parameter": ["Trap depth", "D2 detuning", "Trap frequency: radial", "Trap frequency: axial", "Trap frequency: ratio"],
#         "Symbol": ["U₀", "ΔE", "ω_r / 2π", "ω_z / 2π", "ω_r / ω_z"],
#         "Value": [U0 / h * 1e-6, deltaE * 1e-6, omega_r / (2 * math.pi) * 1e-3, omega_z / (2 * math.pi) * 1e-3, omega_r / omega_z],
#         "Units": ["MHz", "MHz", "kHz", "kHz", ""]
#     })
# else:
data = pd.DataFrame({
    "Parameter": ["Trap depth", "Trap frequency: x", "Trap frequency: y", "Trap frequency: vertical", "Fermi Temperature", "Fermi Velocity"],
    "Symbol": ["U₀", "ω_x / 2π", "ω_y / 2π", "ω_z / 2π", "T_f", "v_F*100"],
    "Value": [U0 / const.k * 1e6, wx, wy, U0, Tf * 1e6, vf],
    "Units": ["μK", "Hz", "Hz", "Hz", "μK" ,  "cm/s"]
})

st.write(data)
