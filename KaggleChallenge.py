#Image hasn't been Divided by 255.0



#!/usr/bin/env python
# coding: utf-8

# # APTOS 2019 Blindness Detection

# # Abstract
# Millions of people suffer from diabetic retinopathy,which is the leading cause of blindness among working aged adults.Currently, Medical technicians travel to rural areas to capture images and then rely on highly trained doctors to review the images and provide diagnosis. Our Aim is to solve this problem by using Machine Learning Techniques to Provide Asia Pacific Tele-Ophthalmology Society (APTOS) with an Algorithim that can classify the Image to one of the following Categories:
# - 0 - No DR
# - 1 - Mild
# - 2 - Moderate
# - 3 - Severe
# - 4 - Proliferative DR
# 
# 
# ![Classification](http://cceyemd.com/wp-content/uploads/2017/08/5_stages.png)
# 
# Image source: http://cceyemd.com/diabetes-and-eye-exams/
# 
# 
# 
# The First Section of this Notebook will do an Exploratory data analysis(EDA) of the APTOS dataset. 
# The second Section of this Notebook will use Different Deep Learning Techniques to Classify The data.
# 

# ### Linux Commands For working with Kaggle Api on Google Cloud
# - !pip install kaggle
# - !mkdir -p .kaggle
# - !cp kaggle.json .kaggle/
# - !ls .kaggle
# - !chmod 600 .kaggle/kaggle.json  # set permission
# - !~/.local/bin/kaggle competitions download -c aptos2019-blindness-detection
# 
# ### Tech Versions
# - Python 3.5.3
# - Tensorflow 2.0

# In[1]:
  
     
#installing dependencies
import zipfile
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import csv
import pathlib
import random
from keras.utils.np_utils import to_categorical
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import RMSprop
import os
from PIL import Image
from time import time
from keras.applications.inception_v3 import InceptionV3
# In[2]:
#Getting Paths of All Images
test_dataframe = pd.read_csv('allCells.csv')
X = test_dataframe['image_id']
Y = test_dataframe['diagnosis']
X = ['data/preprocessed_train/' + i + '.png' for i in X]
image_count = len(X)
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMG_HEIGHT = 299
IMG_WIDTH = 299
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
CLASS_NAMES = np.array(['0.0','1.0','2.0','3.0','4.0'])



Y[:5]
X = tf.data.Dataset.from_tensor_slices([i+'|'+str(j) for i,j in zip(X,Y)])

# In[2]:
#Remove at the end
for f in X.take(5):
  print(get_label(f.numpy()))

# In[2]:

#Function modified from https://www.tensorflow.org/tutorials/load_data/images
def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path,'|')
  # The second to last is the class-directory
  return parts[-1] == CLASS_NAMES, parts[0]

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
  label,img_path = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(img_path)
  img = decode_img(img)
  return img, label

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
X = X.map(process_path, num_parallel_calls=AUTOTUNE)
# In[13]:
#Remove at the end.
for image, label in X.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())
print(type(X))
# In[13]:


# In[14]:
#Spliting DataSet into Train Test and Validation/Dev set.
train_size = int(0.9 * image_count)
val_size = int(0.05 * image_count)
test_size = int(0.05 * image_count)
full_dataset = X
train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)
val_dataset = test_dataset.skip(test_size)
test_dataset = test_dataset.take(test_size)



# In[15]:



def InceptionNetworkV3(input_image):# Input Value
    Inception_network = InceptionV3(First_layer)
     
    
    
    #Add Auxillary branch Here.
    
    flatten =  keras.layers.Flatten()(Inception_network)
    dense_to_output = keras.layers.Dense(64, activation='relu')(flatten)
    output_classes = keras.layers.Dense(num_classes, activation='softmax')(dense_to_output)
    
    model = keras.Model(inputs=First_layer, outputs=output_classes)
    opt =RMSprop(lr=0.5)
    model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model


# In[54]:


model = InceptionNetworkV3(train_dataset.take(0))
tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

# In[ ]:





# In[55]:


model.summary()


# In[20]:





# In[21]:

starttime = time()
history = model.fit(train_images, to_categorical(train_classes), epochs=100 , batch_size=128,callbacks=[tensorboard])
endtime = time()
print((endtime - starttime)/60)
# In[56]:


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

train_loss, train_acc = model.evaluate(test_images,  test_classes, verbose=2)


# In[57]:


print("current models Accuracy On it's self {} ".format(train_acc))


# In[ ]:


tf.version()


# In[ ]:






# %%
