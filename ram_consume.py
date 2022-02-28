import streamlit as st

ram = st.slider(
    label='Select the amount of RAM to consume',
    min_value=0, # 0 mb
    max_value=4096, # 4096mb = 4gb
    value=0,
    step=1
)

megaram = ram * 10000000

str = ' ' * megaram
