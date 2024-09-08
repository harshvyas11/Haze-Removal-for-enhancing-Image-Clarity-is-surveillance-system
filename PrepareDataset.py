# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:36:13 2024

@author: Admin
"""


import cv2
import numpy as np
import random
from scipy import ndimage

import DEHAZE_MODELS

hazy_imgs = np.zeros((55,1200,1600,3))
GT_imgs = np.zeros((55,1200,1600,3))


data_folder = 'DenseHaze_1/'
for i in range(1,56):
    #tmp = np.asarray(cv2.imread(data_folder+str(i).zfill(2)+'_hazy.png'))
    hazy_imgs[i-1,:,:,:] = np.asarray(cv2.imread(data_folder+str(i).zfill(2)+'_hazy.png'), dtype=np.float32)
    GT_imgs[i-1,:,:,:] = np.asarray(cv2.imread(data_folder+str(i).zfill(2)+'_GT.png'), dtype=np.float32)

    
hazy_imgs = hazy_imgs/255.0
GT_imgs = GT_imgs/255.0

train_hazy_imgs = hazy_imgs[:45]

train_GT_imgs = GT_imgs[:45]

test_hazy_imgs = hazy_imgs[45:]

test_GT_imgs = GT_imgs[45:]    

del hazy_imgs
del GT_imgs



def batch_random_crop(img, mask, width, height):
    #assert img.shape[1] >= height
    #assert img.shape[2] >= width
    #assert img.shape[1] == mask.shape[1]
    #assert img.shape[2] == mask.shape[2]
    x = random.randint(0, img.shape[2] - width)
    y = random.randint(0, img.shape[1] - height)
    img = img[:,y:y+height, x:x+width,:]
    mask = mask[:,y:y+height, x:x+width,:]
    return img, mask 


'''   
from dcp_dehaze import dcp_dehaze
#test_hazy_imgs, test_GT_imgs = batch_random_crop(test_hazy_imgs, test_GT_imgs, width=256, height=256)
for i in range(test_hazy_imgs.shape[0]):
    #dcp_dh_img = dcp_dehaze(test_hazy_imgs[i])
    dcp_dh_img = test_hazy_imgs[i]
    #score = DEHAZE_MODELS.SSIM(np.asarray(test_GT_imgs[i],dtype=np.float32), np.asarray(dcp_dh_img,dtype=np.float32))
    im1a = ndimage.zoom(np.asarray(test_hazy_imgs[i],dtype=np.float32), 1)
    im1b = ndimage.zoom(np.asarray(test_GT_imgs[i],dtype=np.float32), 1)
    score1 = DEHAZE_MODELS.SSIM(im1a, im1b)
    
    im1a = ndimage.zoom(np.asarray(test_hazy_imgs[i],dtype=np.float32), 0.5)
    im1b = ndimage.zoom(np.asarray(test_GT_imgs[i],dtype=np.float32), 0.5)
    score2 = DEHAZE_MODELS.SSIM(im1a, im1b)
    
    im1a = ndimage.zoom(np.asarray(test_hazy_imgs[i],dtype=np.float32), 0.25)
    im1b = ndimage.zoom(np.asarray(test_GT_imgs[i],dtype=np.float32), 0.25)
    score3 = DEHAZE_MODELS.SSIM(im1a, im1b)
    
    print('***********')
    print(score1)
    print(score2)
    print(score3)
    
    


exit()
'''



def BatchGeneratorFun(is_validate = False):
    while True:
        batch_size=8
        if is_validate == False:
            idx = np.random.permutation(train_hazy_imgs.shape[0])
            batch_img_arr = train_hazy_imgs[idx[:batch_size]]
            batch_seg_arr = train_GT_imgs[idx[:batch_size]]           
            batch_img_arr, batch_seg_arr = batch_random_crop(batch_img_arr, batch_seg_arr, width=256, height=256)
            
        if is_validate == True:
            idx = np.random.permutation(test_hazy_imgs.shape[0])
            batch_img_arr = test_hazy_imgs[idx[:batch_size]]
            batch_seg_arr = test_GT_imgs[idx[:batch_size]]  
            batch_img_arr, batch_seg_arr = batch_random_crop(batch_img_arr, batch_seg_arr, width=256, height=256)
        
        yield batch_img_arr, batch_seg_arr



model = DEHAZE_MODELS.UNET_COLLECTION('unet_2d')
model.fit(BatchGeneratorFun(is_validate = False),
          validation_data = BatchGeneratorFun(is_validate = True),
          steps_per_epoch=40, 
          validation_steps=2, 
          epochs=200, verbose=True) 