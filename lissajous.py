import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def show_lissajous(x_amp, y_amp, x_freq, y_freq, x_phase, y_phase):
    t =  np.arange(0.00, 4*np.pi, 0.01)
    x = x_amp * np.sin(x_freq * t + x_phase)
    y = y_amp * np.sin(y_freq * t + y_phase)
    fig, ax = plt.subplots()
    ax.plot(x, y, 'b-') # 'b-' means solid blue line
    ax.axis('equal') # equal axis aspect ratio
    return fig

def close_matplotlib_figure(fig):
    plt.close(fig)

default_text = st.empty() # 'empty' element for the default text
default_text.text("Select an option on the left menu") # this text will be shown if no checkboxes are selected on the left menu

if st.sidebar.checkbox(label="Show Lissajous Curve", value=True): # Lissajous Curve will be shown only if checkbox is selected. The checkbox is shown on the sidebar.
    default_text.empty() # clear default text
    st.subheader("Lissajous curve") # add subheader
    x_amp = st.sidebar.slider("Amplitude of the first oscillation", 1, 10, 2)
    y_amp = st.sidebar.slider("Amplitude of the second oscillation", 1, 10, 1)
    x_freq = st.sidebar.slider("Frequency of the first oscillation", 1, 20, 4)
    y_freq = st.sidebar.slider("Frequency of the second oscillation", 1, 20, 7)
    x_phase = st.sidebar.slider("Initial phase of the first oscillation", 0.0, 2*np.pi, 0.0, 0.01)
    y_phase = st.sidebar.slider("Initial phase of the second oscillation", 0.0, 2*np.pi, np.pi/2, 0.01)

    st.pyplot(show_lissajous(x_amp, y_amp, x_freq, y_freq, x_phase, y_phase))

    if st.button("Show animation"):
        lissajous_container = st.empty()
        while y_phase < (y_phase + np.pi):
            with lissajous_container:
                fig = show_lissajous(x_amp, y_amp, x_freq, y_freq, x_phase, y_phase)
                st.pyplot(fig)
                close_matplotlib_figure(fig)
            y_phase += 0.25
