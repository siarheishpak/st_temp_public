import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

def show_lissajous(x_amp, y_amp, x_freq, y_freq, x_phase, y_phase):
    t =  np.arange(0.00, 2*np.pi, 0.01)
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
    y_phase = st.sidebar.slider("Initial phase of the second oscillation", 0.0, 2*np.pi, np.pi/2, 0.01, key='y_phase_key')

    st.pyplot(show_lissajous(x_amp, y_amp, x_freq, y_freq, x_phase, y_phase))

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            show = st.button("Show animation")
            animation_speed_factor = st.number_input(
                label='Set the speed of animation',
                value=1,
                min_value=1,
                max_value=10,
                step=1,
                help='The higher value, the quicker animation'
            )

    if show:
        with col2:
            hide = st.button("Hide animation") # clicking the button will re-render the page and therefore remove the animation
        y_phase_i = y_phase
        progress_value = 0.0
        progress_bar = st.progress(progress_value) # to show progress bar
        lissajous_container = st.empty()
        period = (2 * np.pi) / math.gcd(x_freq, y_freq) # period of the lissajous curve is a period of animation
        factor = 0.001 * animation_speed_factor # this var is related to the speed of the animation
        step = factor * period # step of phase changing in animation
        while y_phase_i < (y_phase + period):
            progress_bar.progress(progress_value)
            with lissajous_container:
                fig = show_lissajous(x_amp, y_amp, x_freq, y_freq, x_phase, y_phase_i)
                st.pyplot(fig)
                close_matplotlib_figure(fig)
            y_phase_i += step
            progress_value = round(progress_value + factor, 3)


if checkbox('line_chart'):
    t =  np.arange(0.00, 2*np.pi, 0.01)
    x = []
    y = []
    for i in range(len(t)):
        x.append(x_amp * np.sin(x_freq * t + x_phase))
        y.append(y_amp * np.sin(y_freq * t + y_phase))
    df = pd.DataFrame(y, index=x)
    st.line_chart(df)
