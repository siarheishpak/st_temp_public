import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(256)
sp = np.fft.fft(np.sin(t))
freq = np.fft.fftfreq(t.shape[-1])
# plt.plot(freq, sp.real, freq, sp.imag)
#fig = plt.plot(freq, sp.real, freq, sp.imag)
#st.pyplot(fig)
print(sp)
#st.write(freq)
