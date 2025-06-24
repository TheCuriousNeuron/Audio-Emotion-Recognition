#!/usr/bin/env python
# coding: utf-8

# In[3]:


# In[5]:


import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
import librosa
import pickle


# In[7]:


st.set_page_config(page_title="Emotion Recognition",page_icon="🎵",layout="centered")


# In[9]:


def load_model_and_preprocessors():
    json_file=open('emotion_model.json','r')
    loaded_model_json=json_file.read()
    json_file.close()
    loaded_model=model_from_json(loaded_model_json)
    loaded_model.load_weights("emotion_model.weights.h5")
    
    with open('Standard_Scaler.pickle','rb') as f:
        scaler=pickle.load(f)
    with open('One_Hot_Encoding.pickle','rb') as f1:
        encoder=pickle.load(f1)
    
    return loaded_model,scaler,encoder


# In[11]:


model,scaler,encoder=load_model_and_preprocessors()


# In[13]:


def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)
    
def rmse(data,frame_length=2048,hop_length=512):
    rmse=librosa.feature.rms(y=data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(rmse)
    
def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
    mfcc=librosa.feature.mfcc(y=data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

def extract_features(data,sr=22050,frame_length=2048,hop_length=512):
    result=np.array([])
    result=np.hstack((result,zcr(data,frame_length,hop_length),rmse(data,frame_length,hop_length),mfcc(data,sr,frame_length,hop_length)))
    
    return result


# In[15]:


def get_predict_feat(path):
    d,s_rate= librosa.load(path,duration=2.5, offset=0.6)
    res=extract_features(d)
    result=np.array(res)
    result=np.reshape(result,newshape=(1,2376))
    i_result=scaler.transform(result)
    final_result=np.expand_dims(i_result,axis=2)
    
    return final_result


# In[17]:


emotions1={1:'Neutral', 2:'Calm', 3:'Happy', 4:'Sad', 5:'Angry', 6:'Fear', 7:'Disgust',8:'Surprise'}
def prediction(path1):
    res=get_predict_feat(path1)
    predictions=model.predict(res)
    y_pred = encoder.inverse_transform(predictions)
    return (y_pred[0][0])


# In[19]:

# In[21]:


st.title("🎵 Audio Emotion Recognition")
st.markdown("Upload an audio file to detect the emotion in speech")

uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'flac'],help="Supported formats: WAV, MP3, FLAC")


# In[23]:


if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')


# In[29]:


if st.button("🔍 Analyze Emotion", type="primary"):
        with st.spinner('Analyzing audio...'):
            try:
                emotion = prediction(uploaded_file)
                st.success(f"**Detected Emotion: {emotion.upper()}**")
                    
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")


# In[31]:


st.sidebar.markdown("About")
st.sidebar.info("This app uses a CNN model with ZCR, RMSE, and MFCC features for emotion recognition.")


# In[33]:


st.sidebar.markdown("Supported Emotions:")
emotions_list = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise']
for emotion in emotions_list:
    st.sidebar.write(f"• {emotion}")


# In[ ]:





# In[ ]:




