from scipy.misc import imread # using scipy's imread
import cv2
import numpy as np
import matplotlib.pyplot as plt


###############################################################################
# Utility functions
###############################################################################

def boundaries(binarized,axis):
    # variables named assuming axis = 0; algorithm valid for axis=1
    # [1,0][axis] effectively swaps axes for summing
    rows = np.sum(binarized,axis = [1,0][axis]) > 0
    rows[1:] = np.logical_xor(rows[1:], rows[:-1])
    change = np.nonzero(rows)[0]
    ymin = change[::2]
    ymax = change[1::2]
    height = ymax-ymin
    too_small = 10 # real letters will be bigger than 10px by 10px
    ymin = ymin[height>too_small]
    ymax = ymax[height>too_small]
    return zip(ymin,ymax)


def separate(img):
    orig_img = img.copy()
    pure_white = 255.
    white = np.max(img)
    black = np.min(img)
    thresh = (white+black)/2.0
    binarized = img<thresh
    row_bounds = boundaries(binarized, axis = 0) 
    cropped = []
    for r1,r2 in row_bounds:
        img = binarized[r1:r2,:]
        col_bounds = boundaries(img,axis=1)
        rects = [r1,r2,col_bounds[0][0],col_bounds[0][1]]
        cropped.append(np.array(
                orig_img[rects[0]:rects[1],rects[2]:rects[3]]/pure_white))
    return cropped

###############################################################################
# Recognizing professor's handwiring
###############################################################################

## Create target and data
    
# Assembly target
    
target_a = np.full((23,), 0)
target_b = np.full((23,), 1)
target_c = np.full((23,), 2)
target = np.append(np.append(target_a, target_b), target_c)

# Read columns of images in grayscale
column_a = imread('a.png', flatten = True)
column_b = imread('b.png', flatten = True)
column_c = imread('c.png', flatten = True)

# Separate columns into arrays of cropped images
imgs_a = separate(column_a)
imgs_b = separate(column_b)
imgs_c = separate(column_c)

# Resize images to 5px
resized_a = []

for img in imgs_a:
    resized_a.append(cv2.resize(
            img,
            (5, 5),
            interpolation=cv2.INTER_AREA))

resized_b = []
for img in imgs_b:
    resized_b.append(cv2.resize(
            img,
            (5, 5),
            interpolation=cv2.INTER_AREA))

resized_c = []
for img in imgs_c:
    resized_c.append(cv2.resize(
            img,
            (5, 5),
            interpolation=cv2.INTER_AREA))    

# 8x8 images of letters a, b, c
images_abc = np.array(resized_a + resized_b + resized_c)

# Convert to (samples, feature) matrix by flattening
num = len(images_abc)
data = images_abc.reshape((num, -1))
