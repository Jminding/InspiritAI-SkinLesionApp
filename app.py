import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
import os
model = load_model("model",compile=False)
PAGE_CONFIG = {"page_title":"Skin Cancer Diagnosis","page_icon":"https://cdn.upload.systems/uploads/IXr6V3Az.png","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)
st.sidebar.title("Skin Cancer Diagnosis with AI and ML")
st.title("Skin Cancer Diagnosis with AI and ML")
menu = ["About","Get Diagnosed"]
choice = st.sidebar.selectbox('Page Selection', menu)
if choice == 'Get Diagnosed':
  st.header("Get Diagnosed!")
  st.write("1. Upload a picture of your skin lesion:")
  uploaded_file = st.file_uploader("Upload a picture",type=["png","jpg","jpeg","heic", "heif", "hevc"])
  st.write("2. Receive your diagnosis!")
  if uploaded_file is not None and st.button("Receive Your Diagnosis"):
    try:  
      file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
      image = cv2.imdecode(file_bytes, 1)
      st.image(image, caption="Your Image", channels="BGR")

      small = cv2.resize(image, (75, 100)) #YOUR CODE HERE: specify image, dimensions
      #gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

      gray_flat = np.reshape(small, (1, 75, 100, 3))
      prediction = model.predict(gray_flat).all()
      if prediction == 1:
        st.error('Skin Cancer Detected')
      else:
        st.success('Skin Cancer Not Detected')
    except Exception as e:
        st.error("Something went wrong!  Please try again later or contact the devs.")
        st.exception(e)
elif choice == 'About':
  st.subheader("About this project")
  st.info("This project was made in the Inspirit AI program to (somewhat) accurately diagnose skin lesions as being cancerous or noncancerous.")
  st.info("Would you rather go to the doctor, wait a while, and receive a diagnosis, or simply take a picture and receive a diagnosis?  That's what our Machine Learning tool does - it analyzes pictures of skin lesion and determines whether or not they are cancerous.")
  st.sidebar.warning("Disclaimer: We'd recommend still going to see a doctor after this.  If you are worried about your skin lesions, don't just sit there - please go and get a diagnosis from a doctor (and our website if you wish).")
  st.write("Go get a diagnosis!  Head to the sidebar on the left and select \"Get Diagnosed\" from the page selection.")
  st.empty()
  st.container()
  col1, col2, col3 = st.columns(3)
  col2.image("https://cdn.upload.systems/uploads/RKa2oIVM.jpeg")
