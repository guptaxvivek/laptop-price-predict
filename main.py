import pickle
import streamlit as st
import numpy as np


df = pickle.load(open('df.pkl','rb'))
pipe = pickle.load(open('pipe.pkl','rb'))

st.title("LAPTOP PRICE PREDICTOR")

st.image("laptop.jpg")

#Enter Company
company = st.selectbox("Company",df["Company"].unique())

#Enter Type
type = st.selectbox("Type",df["TypeName"].unique())

#Enter Ram
ram = st.selectbox("RAM (in GBs)",[2,4,6,8,12,16,24,32,64])

#Enter Weight
weight = st.number_input("Weight(in kgs)")

#Touchscreen
touch = st.selectbox("Touchscreen",["No","Yes"])

#Ips
ips = st.selectbox("IPS Display",["No","Yes"])

#Resolution
res = st.selectbox("Screen Resolution",["1366x768","1600x900","1920x1080","2304x1440","2560x1440","2560x1600","2880x1800","3000x2000","3200x1800","3840x2160"])

#Screen Size
size = st.number_input("Screen Size(in inches)")

#Cpu
cpu = st.selectbox("CPU",df["Cpu Brand"].unique())

#storage
hdd = st.selectbox("Hard Disk(HDD)",[0,64,128,256,512,1024,2048])

#ssd
ssd = st.selectbox("SSD",[0,128,256,512,1024])

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
    st.title("The Price for laptop should be: ₹{:.2f}".format(np.exp(pipe.predict(feature)[0])))