import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def show_lissajous(x_amp, y_amp, x_freq, y_freq, x_phase, y_phase):
    t =  np.arange(0.00, 4*np.pi, 0.01)
    x = x_amp * np.sin(x_freq * t + x_phase)
    y = y_phase * np.sin(y_freq * t + y_phase)
    fig, ax = plt.subplots()
    ax.plot(x, y, 'b-')
    return fig

x_amp = st.slider("Amplitude of the first oscillation", 1, 10, 1)
y_amp = st.slider("Amplitude of the second oscillation", 1, 10, 1)
x_freq = st.slider("Frequency of the first oscillation", 1, 20, 3)
y_freq = st.slider("Frequency of the second oscillation", 1, 20, 7)
x_phase = st.slider("Initial phase of the first oscillation", 0.0, 2*np.pi, 0.0, 0.01)
y_phase = st.slider("Initial phase of the second oscillation", 0.0, 2*np.pi, np.pi/6, 0.01)
st.pyplot(show_lissajous(x_amp, y_amp, x_freq, y_freq, x_phase, y_phase))
