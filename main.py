import pickle
import time

import streamlit as st
import numpy as np
import sklearn


df = pickle.load(open('df.pkl','rb'))
pipe = pickle.load(open('pipe.pkl','rb'))

st.title("LAPTOP PRICE PREDICTOR")

st.image("laptop.jpg")

col1,col2 = st.columns(2)
col3,col4 = st.columns(2)
col5,col6 = st.columns(2)
col7,col8 = st.columns(2)
col9,col10 = st.columns(2)
col11,col12 = st.columns(2)


with col1:
    #Enter Company
    company = st.selectbox("Company",df["Company"].unique())

with col2:
    #Enter Type
    type = st.selectbox("Type",df["TypeName"].unique())

with col3:
    #Enter Ram
    ram = st.selectbox("RAM (in GBs)",[2,4,6,8,12,16,24,32,64])
with col4:
    #Enter Weight
    weight = st.number_input("Weight(in kgs)")

with col5:
    #Touchscreen
    touch = st.selectbox("Touchscreen",["No","Yes"])

with col6:
    #Ips
    ips = st.selectbox("IPS Display",["No","Yes"])

with col7:
    #Resolution
    res = st.selectbox("Screen Resolution",["1366x768","1600x900","1920x1080","2304x1440","2560x1440","2560x1600","2880x1800","3000x2000","3200x1800","3840x2160"])

with col8:
    #Screen Size
    size = st.number_input("Screen Size(in inches)")

with col9:
    #Cpu
    cpu = st.selectbox("CPU",df["Cpu Brand"].unique())

with col10:
    #storage
    hdd = st.selectbox("Hard Disk(HDD)",[0,64,128,256,512,1024,2048])

with col11:
    #ssd
    ssd = st.selectbox("SSD",[0,128,256,512,1024])

with col12:
    #gpu
    gpu = st.selectbox("GPU Brand",df["Gpu brand"].unique())

#os
os = st.selectbox("Operating System",df["OS"].unique())



if st.button("Predict Price"):
    ResX = int(res.split('x')[0])
    ResY = int(res.split('x')[1])
    ppi = ((ResX ** 2) + (ResY ** 2)) ** 0.5 / size

    if touch == "No":
        touch = 0
    else:
        touch = 1

    if ips == "No":
        ips = 0
    else:
        ips = 1

    feature = np.array([company, type, ram, weight, touch, ips, ppi, cpu, hdd, ssd, gpu, os]).reshape(1,12)

    with st.spinner(text="In progress..."):
        time.sleep(2)
        pred = np.exp(pipe.predict(feature)[0])

    st.header("The Price for laptop should be: ₹{:.2f}".format(pred))