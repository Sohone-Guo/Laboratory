# -*- coding: utf-8 -*-
# This code is for practice CNN
# There are severial steps base on the CNN, 
# Convolution; Max pooling; DNN

import numpy as np
from PIL import Image
import CNN_sohone.convoluation

# Read the image
img = np.asarray(Image.open("Data/image.png"))

# Initial the CNN
CNN = CNN_sohone.convoluation.CNN()
CNN.trainning(img)
