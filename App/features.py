#Test ảnh
from keras.applications import VGG16
from keras.applications import imagenet_utils
import cv2
from keras.preprocessing.image import img_to_array
from random import choices
import numpy as np
import random
import os
from os import walk

def testImage(path):
  model1 = VGG16(weights='imagenet', include_top=False)
  img = cv2.imread(path)
  img = cv2.resize(img,(224,224))
  img = img_to_array(img)
  img = np.expand_dims(img, 0)
  img = imagenet_utils.preprocess_input(img)
  imgfeat = model1.predict(img)
  imgfeat = imgfeat.reshape((imgfeat.shape[0], 512*7*7))
  return imgfeat

def nameFlower(pred):
  if(pred[0]==0):
    return "Hoa Cúc"
  if(pred[0]==1):
    return "Hoa Bồ Công Anh"
  if(pred[0]==2):
    return "Hoa Hồng"
  if(pred[0]==3):
    return "Hoa Hướng Dương"
  if(pred[0]==4):
    return "Hoa Tuy Líp"


def random_images(name):
  f = []
  basedir = os.path.abspath(os.path.dirname(__file__))
  path = os.path.join(basedir, 'static/img', name)
  for (dirpath, dirnames, filenames) in walk(path):
    f.extend(filenames)
  idx = random.randint(0, len(f)-4)
  return list(map(lambda image_name: name + '/' +image_name, f[idx:idx+4]))

def choose_similar_files(pred):
  if(pred[0]==0):
    return random_images('daisy')
  if(pred[0]==1):
    return random_images('dandelion')
  if(pred[0]==2):
    return random_images('rose')
  if(pred[0]==3):
    return random_images('sunflower')
  if(pred[0]==4):
    return random_images('tulip')



def input_CNN(path):
  img = cv2.imread(path)
  img = cv2.resize(img,(224,224))
  img = img_to_array(img)
  img = np.expand_dims(img, 0)
  img = imagenet_utils.preprocess_input(img)
  return img


import joblib


# model_CNN=joblib.load('fine_tuning.pkl')
# img=input_CNN('rose1.jpg')

# pred=model_CNN.predict(img)
# rounded_pred=np.argmax(pred, axis=1)
# print('Giá trị dự đoán model CNN: ',nameFlower(rounded_pred))
# imgfeat=testImage("rose1.jpg")

# preds2 = svm_linear.predict(imgfeat.reshape(1,-1))
