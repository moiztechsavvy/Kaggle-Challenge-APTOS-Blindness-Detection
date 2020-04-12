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
import matplotlib.pyplot as plt
import cv2
import csv
import pathlib
import random
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import models,layers
import os
#This Library wasn't supported with Tensorflow2.0
#import tensorflow_addons as tfa
from PIL import Image
from time import time
from tensorflow.keras.applications.inception_v3 import InceptionV3


# In[3]
# 
def kappa_loss(y_pred, y_true, y_pow=2, eps=1e-10, N=5, bsize=256, name='kappa'):
  """A continuous differentiable approximation of discrete kappa loss.
      Args:
          y_pred: 2D tensor or array, [batch_size, num_classes]
          y_true: 2D tensor or array,[batch_size, num_classes]
          y_pow: int,  e.g. y_pow=2
          N: typically num_classes of the model
          bsize: batch_size of the training or validation ops
          eps: a float, prevents divide by zero
          name: Optional scope/name for op_scope.
      Returns:
          A tensor with the kappa loss."""
  with tf.name_scope(name):
      y_true = tf.cast(y_true,dtype='float')
      repeat_op = tf.cast(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]), dtype='float')
      repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
      weights = repeat_op_sq / tf.cast((N - 1) ** 2, dtype='float')

      pred_ = y_pred ** y_pow
      try:
          pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
      except Exception:
          pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))

      hist_rater_a = tf.reduce_sum(pred_norm, 0)
      hist_rater_b = tf.reduce_sum(y_true, 0)

      conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)

      nom = tf.reduce_sum(weights * conf_mat)
      denom = tf.reduce_sum(weights * tf.matmul(
          tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /
                            tf.cast(bsize, dtype='float'))
      return nom / (denom + eps)
# In[2]:
#Getting Paths of All Images
test_dataframe = pd.read_csv('allCells.csv')
X = test_dataframe['image_id']
Y = test_dataframe['diagnosis']
X = ['data/preprocessed_train/' + i + '.png' for i in X if ".png" not in i]
image_count = len(X)
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMG_HEIGHT = 299
IMG_WIDTH = 299
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
CLASS_NAMES = np.array(['0.0','1.0','2.0','3.0','4.0'])



Y[:5]
X = tf.data.Dataset.from_tensor_slices([i+'|'+str(j) for i,j in zip(X,Y)])
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
  return img/255, label
# In[2]:
#Remove at the end
for f in X.take(5):
  print(get_label(f.numpy()))

# In[2]:

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
X = X.map(process_path, num_parallel_calls=AUTOTUNE)
# In[13]:
#Remove at the end.
for image, label in X:
  print("Image shape: ", image.numpy())
  print("Label: ", label.numpy())
print(type(X))

# In[13]:


# In[14]:
#Spliting DataSet into Train Test and Validation/Dev set.
train_size = int(0.9 * image_count)
val_size = int(0.05 * image_count)
test_size = int(0.05 * image_count)
full_dataset = X
train_dataset = full_dataset.take(train_size).shuffle(1000)
test_dataset = full_dataset.skip(train_size)
val_dataset = test_dataset.skip(test_size)
test_dataset = test_dataset.take(test_size)



# In[15]:

print(test_dataset)

def InceptionNetworkV3():# Input Value
  input_shape = (299,299,3)
  model = models.Sequential()
  model.add(InceptionV3(input_shape=input_shape, weights=None, include_top=False))
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(5))
  model.compile(optimizer=tf.compat.v1.train.RMSPropOptimizer(0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  return model
#kappa_loss optimizer

# In[54]:


model = InceptionNetworkV3()
tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
model.summary()


# In[ ]:
type(train_dataset)
# iterable_ds = iter(train_dataset.batch(100).repeat(100))


# In[21]:
#sess = tf.Session()
history = model.fit(train_dataset.batch(100).repeat(100),shuffle=True , epochs=100 ,callbacks=[tensorboard])
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
xs.shape


# In[ ]:






# %%
