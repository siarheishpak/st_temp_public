import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def show_lissajous():
    t =  np.arange(0.00, 2*np.pi, 0.01)
    x = np.sin(2 * t + (np.pi / 6))
    y = 3 * np.sin(7 * t)
    fig, ax = plt.subplots()
    ax.plot(x, y, 'b-')
    return fig

st.pyplot(show_lissajous())
