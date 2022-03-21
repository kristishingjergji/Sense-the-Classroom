#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fee
import cv2
import matplotlib.pyplot as plt
import time
start_time = time.time()


# In[4]:


model_folder = "/Users/kristshingjergji/Documents/PhD/Experiment_1/models"
#model_folder = input("Folder of model: ")
predictor = fee.openFiles(model_folder)[0]
image_folder = '/Users/kristshingjergji/Desktop/images_datasets/images/image.jpg'
image = cv2.imread(image_folder)


# In[5]:


plt.imshow(image)


# In[6]:


landmarks, rects = fee.face_landmarks(image, predictor = predictor)


# In[7]:


rotatedImage, rotatedLandmarks = fee.align_from_landmarks(image, predictor = predictor)


# In[8]:


resizedCroppedImage, resizedCroppedLandmarks = fee.crop_resize(image, 112, predictor = predictor)


# In[9]:


feature_cbd = fee.hogsLandmarks(resizedCroppedImage, resizedCroppedLandmarks, model_folder)


# In[10]:


AUs = fee.AUdetection(feature_cbd, model_folder)


# In[11]:


fee.plotAUs(image, AUs, landmarks,rects)
plt.show()


print('Success!')
print("--- %s seconds ---" % (time.time() - start_time))




