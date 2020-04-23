# In[3]
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
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from time import time
from tensorflow.keras.applications.inception_v3 import InceptionV3


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128
IMG_HEIGHT = 299
IMG_WIDTH = 299
EPOCHS = 100
CLASS_NAMES = np.array(['0.0','1.0','2.0','3.0','4.0'])
# In[3]
#
def kappa_loss(y_pred, y_true, y_pow=2, eps=1e-10, N=5, bsize=BATCH_SIZE, name='kappa'):
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
test_dataframe = pd.read_csv('/home/awall03/Datafile/allCells.csv')
X = test_dataframe['image_id']
y = test_dataframe['diagnosis']
X = ['/home/awall03/Datafile/data/preprocessed_train/' + i + '.png' for i in X if ".png" not in i]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
train_image_count = len(X_train)
STEPS_PER_EPOCH = np.ceil(train_image_count/BATCH_SIZE)
print(train_image_count,STEPS_PER_EPOCH)

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

def datasetbuilder(x,y,batchsize):
  ds = tf.data.Dataset.from_tensor_slices([i+'|'+str(j) for i,j in zip(x,y)])
  ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
  ds = ds.shuffle(1024)
  ds = ds.batch(batchsize,drop_remainder=True)
  ds = ds.repeat()
  ds = ds.prefetch(AUTOTUNE)
  ds = ds.map(lambda img, label: (tf.image.convert_image_dtype(img,dtype=tf.float32), label))
  return ds



# In[2]:
# Training Dataset
train_dataset = datasetbuilder(X_train,y_train,BATCH_SIZE)

#Test Dataset
test_dataset = datasetbuilder(X_test,y_test,BATCH_SIZE)

# In[15]:

def InceptionNetworkV3():# Input Value
 input_shape = (299,299,3)
 model = models.Sequential()
 model.add(InceptionV3(input_shape=input_shape, weights=None, include_top=False))
 model.add(layers.Flatten())
 model.add(layers.Dense(5))

 model.compile(optimizer=tf.compat.v1.train.RMSPropOptimizer(0.01),
             loss='categorical_crossentropy',
             metrics=['accuracy',kappa_loss])
 return model
#kappa_loss optimizer

# In[54]:


model = InceptionNetworkV3()
tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
model.summary()


# In[ ]:

history = model.fit(train_dataset,shuffle=True , steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS ,callbacks=[tensorboard])
# In[56]: