#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
import joblib 


# In[2]:


def preProcessText(text):
    for doc in text:
        doc = re.sub(r"\\n", "", doc)
        doc = re.sub(r"\W", " ", doc) #remove non words char
        doc = re.sub(r"\d"," ", doc) #remove digits char
        doc = re.sub(r'\s+[a-z]\s+', "", doc) # remove a single char
        doc = re.sub(r'^[a-z]\s+', "", doc) #remove a single character at the start of a document
        doc = re.sub(r'\s+', " ", doc)  #replace an extra space with a single space
        doc = re.sub(r'^\s', "", doc) # remove space at the start of a doc
        doc = re.sub(r'\s$', "", doc) # remove space at the end of a document
    return doc.lower()


# In[3]:


model = joblib.load('model.pkl')


# In[4]:


model_vocab = joblib.load('model_vocab.pkl')


# In[7]:


def change(st):
  sent = [w for w in st.split() if w in model_vocab]
  return sent


# In[8]:


def similarity_check(text1,text2):
  text1 = change(text1)
  text2 = change(text2)
  val = model.wv.n_similarity(text1,text2)
  return val


# In[11]:


from flask import Flask, jsonify,request
import json
from flask_cors import CORS
import time
app = Flask(__name__)
CORS(app)
@app.route("/")
def main():
    return 'Hello from get request'

@app.route('/api/sent',methods=['POST'])
def add():
    text1 = request.json["text1"]
    text2 = request.json["text2"]
    output_dict = {}
#     print(type(similarity_check(text1,text2)))
    output_dict['Similarity_percent'] = str(similarity_check(text1,text2))
#     print(output_dict)
    return jsonify(output_dict)

if __name__ == '__main__':
    app.run(host = "0.0.0.0",port = 8080,debug = True)


# In[ ]:




