import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

t = np.arange(32)
sp = np.fft.fft(np.sin(t))
freq = np.fft.fftfreq(t.shape[-1])
# plt.plot(freq, sp.real, freq, sp.imag)
fig, ax = plt.subplots()
ax.plot(t, sp, 'b-') # 'b-' means solid blue line
ax.axis('equal') # equal axis aspect ratio
st.pyplot(fig)

data = pd.DataFrame(t)
st.line_chart(data)
