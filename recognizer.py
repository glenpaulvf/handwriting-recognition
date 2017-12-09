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

# 5x5 images of letters a, b, c
images_abc = np.array(resized_a + resized_b + resized_c)

# Convert to (samples, feature) matrix by flattening
num = len(images_abc)
data = images_abc.reshape((num, -1))


###############################################################################
# Recognizing own handwiring
###############################################################################

## Create target and data
    
# Assembly target, respective to alphabet

me_target_glen = [7, 12, 5, 14]*5 # collated
me_target = np.array(me_target_glen)

# Read columns of images in grayscale
me_column_g = imread('me_g.png', flatten = True)
me_column_l = imread('me_l.png', flatten = True)
me_column_e = imread('me_e.png', flatten = True)
me_column_n = imread('me_n.png', flatten = True)

# Separate columns into arrays of cropped images
me_imgs_g = separate(me_column_g)
me_imgs_l = separate(me_column_l)
me_imgs_e = separate(me_column_e)
me_imgs_n = separate(me_column_n)

# Resize images to 5px
me_resized_g = []
for img in me_imgs_g:
    me_resized_g.append(cv2.resize(
            img,
            (5, 5),
            interpolation=cv2.INTER_AREA))

me_resized_l = []
for img in me_imgs_l:
    me_resized_l.append(cv2.resize(
            img,
            (5, 5),
            interpolation=cv2.INTER_AREA))
    
me_resized_e = []
for img in me_imgs_e:
    me_resized_e.append(cv2.resize(
            img,
            (5, 5),
            interpolation=cv2.INTER_AREA))
    
me_resized_n = []
for img in me_imgs_n:
    me_resized_n.append(cv2.resize(
            img,
            (5, 5),
            interpolation=cv2.INTER_AREA))

# Collate images
me_images_collate = []
for x in zip(me_resized_g, me_resized_l, me_resized_e, me_resized_n):
    for y in x:
        me_images_collate.append(y)

# 5x5 images of letters g, l, e, n
images_glen = np.array(me_images_collate)

# Convert to (samples, feature) matrix by flattening
num = len(images_glen)
data = images_glen.reshape((num, -1))


## Partition data and target into training and test sets

def partition(data, target, p):
    '''
    Partition data and target for training
    Remaining data and target for testing
    
    Since data and target are not ordered (they're collated),
    we simply select the first N samples, where N = p * len of data/target.
    '''
    
    slice_index = len(data) * p
    
    train_data = data[:int(slice_index)]
    train_target = target[:int(slice_index)]
    test_data = data[int(slice_index):]
    test_target = target[int(slice_index):]

    return train_data, train_target, test_data, test_target
