import tensorflow as tf
import numpy as np

IMG_HEIGHT = 299
IMG_WIDTH = 299
CLASS_NAMES = np.array(['0.0','1.0','2.0','3.0','4.0'])
AUTOTUNE = tf.data.experimental.AUTOTUNE


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
  ds = ds.batch(batchsize,drop_remainder=True)
  ds = ds.repeat()
  ds = ds.prefetch(AUTOTUNE)
  ds = ds.map(lambda img, label: (tf.image.convert_image_dtype(img,dtype=tf.float32), label))
  return ds