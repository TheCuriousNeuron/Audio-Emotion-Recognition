#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tensorflow.keras.models import model_from_json
json_file=open('emotion_model.json', 'r')
loaded_model_json=json_file.read()
json_file.close()
loaded_model=model_from_json(loaded_model_json)
loaded_model.load_weights("emotion_model.weights.h5")


# In[4]:


import pickle
with open('Standard_Scaler.pickle','rb') as f:
    scaler=pickle.load(f)
    
with open('One_Hot_Encoding.pickle','rb') as f1:
    encoder=pickle.load(f1)


# In[20]:


import librosa
import numpy as np


# In[38]:


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


# In[40]:


def get_predict_feat(path):
    d,s_rate= librosa.load(path,duration=2.5, offset=0.6)
    res=extract_features(d)
    result=np.array(res)
    result=np.reshape(result,newshape=(1,2376))
    i_result=scaler.transform(result)
    final_result=np.expand_dims(i_result,axis=2)
    
    return final_result


# In[42]:


emotions1={1:'Neutral', 2:'Calm', 3:'Happy', 4:'Sad', 5:'Angry', 6:'Fear', 7:'Disgust',8:'Surprise'}
def prediction(path1):
    res=get_predict_feat(path1)
    predictions=loaded_model.predict(res)
    y_pred = encoder.inverse_transform(predictions)
    print(y_pred[0][0])


# In[44]:


prediction("Audio_Combined_Actors_01-24/Actor_01/03-01-03-01-01-01-01.wav")


# In[ ]:




