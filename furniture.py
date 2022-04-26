import streamlit as st

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental import preprocessing

from PIL import Image, ImageOps
from cv2 import cv2

import pandas as pd
import numpy as np

classifier_model = 'efficientnet_furniture.h5'
model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})

st.title('Welcome to the furniture classifier!')

st.markdown("This dashboard takes your image of a chair/coffee table/dresser and makes a prediction about which 1 of those 3x furniture categories it belongs to. More furniture types will be added at a later date...")

with st.expander("Click here for more details about how this model was built"):
        st.write("""This Multiclass Classification model leverages the EfficientNet-B0 Convolutional Neural Network (CNN) to convert images into grids of numbers, which it then scans to discover patterns.""") 
        st.write("""EfficientNet-B0 is a State Of The Art (SOTA) model trained on more than a million images from the ImageNet database (a large visual database designed for use in visual object recognition software research). The network can classify images into one thousand different object categories, including furniture. Woot!""")
        st.write("""The model was trained & tested on 1,000+ images (approx. 360 for each of the 3 furniture types.""")

@st.cache

def import_and_predict(image_data, model):

        size = (256,256)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]    
        
        prediction = model.predict(img_reshape)

        return prediction 
    
file = st.file_uploader("Please upload your picture...", type=["png","jpg","jpeg"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, width=200)
    prediction = import_and_predict(image, model)
    st.write("Here's the model's prediction (expressed as a percentage)...")
  
    df = pd.DataFrame(prediction, columns = ['Chair probability (%)','Coffee Table probability (%)','Dresser probability (%)'])
    percent = df*100
    
    st.write(percent)

 