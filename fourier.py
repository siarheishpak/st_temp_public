import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

t = np.arange(-np.pi, np.pi, 2*np.pi/256)
f = np.cos(2*np.pi*3*t)*np.exp(-np.pi*t*t)
sp = np.fft.fft(f)
real = sp.real
imag = sp.imag
length = np.sqrt(real * real + imag * imag)
freq = np.fft.fftfreq(t.shape[-1])
# plt.plot(freq, sp.real, freq, sp.imag)
fig, ax = plt.subplots()
ax.plot(freq, sp, 'b-') # 'b-' means solid blue line
# ax.axis('equal') # equal axis aspect ratio
st.pyplot(fig)
