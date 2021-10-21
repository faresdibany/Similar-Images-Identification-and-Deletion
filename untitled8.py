# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 13:33:42 2021

@author: Dell
"""
import numpy as np 
import scipy as sp 
import skimage 
import cv2
import imutils 
from skimage import filters
from skimage import io, img_as_float
from skimage.color import rgb2gray
import os 
import functools 
from functools import partial
import operator 
from operator import is_not
# import image_similarity_measures
# from image_similarity_measures.quality_metrics import rmse, ssim, sre

#to draw background to image, so one can identify the specific image to be compared
def draw_color_mask(img, borders, color=(0, 0, 0)):

    h = img.shape[0]

    w = img.shape[1]

    x_min = int(borders[0] * w / 100)

    x_max = w - int(borders[2] * w / 100)

    y_min = int(borders[1] * h / 100)

    y_max = h - int(borders[3] * h / 100)

    img = cv2.rectangle(img, (0, 0), (x_min, h), color, -1)

    img = cv2.rectangle(img, (0, 0), (w, y_min), color, -1)

    img = cv2.rectangle(img, (x_max, 0), (w, h), color, -1)

    img = cv2.rectangle(img, (0, y_max), (w, h), color, -1)

    return img

#blurs the image, and turns it grayscale, for easier comparison and contour finding.
def preprocess_image_change_detection(img, gaussian_blur_radius_list=None, black_mask=(5, 10, 5, 0)):

    gray = img.copy()

    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)#turn image to grayscale

    if gaussian_blur_radius_list is not None:

        for radius in gaussian_blur_radius_list:

            gray = cv2.GaussianBlur(gray, (radius, radius), 0)

    gray = draw_color_mask(gray, black_mask)

    return gray

#finds the contour and processes the image, so one can see the exact difference between images
#in terms of pixels, and their difference in values. 
def compare_frames_change_detection(prev_frame, next_frame, min_contour_area):

    frame_delta = cv2.absdiff(prev_frame, next_frame)
    #80-255 scale for threshold was chosen, as in many images, they are duplicates,...
    #however some lighting affects the comparison, so the lower limit had to be raised
    thresh = cv2.threshold(frame_delta, 80, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,

                            cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    score = 0

    res_cnts = []

    for c in cnts:

        if cv2.contourArea(c) < min_contour_area:

            continue
        
        res_cnts.append(c)

        score += cv2.contourArea(c)
        
    return score, res_cnts, thresh
#defined function to use the above functions for preprocessing and comparison
def compare_images_and_find_similar(img1,img2):
    #after many trials, gaussian blur with kernel size 7 was the most optimum, with regards
    #to accuracy in comparison. For the black mask, the top border is large, due to the 
    #fact that the ceiling of the garage takes a big part of the picture. 
    img1 = preprocess_image_change_detection(img1,[7],(5,30,5,0))
    img2 = preprocess_image_change_detection(img2,[7],(5,30,5,0))
    similarity_scoring = compare_frames_change_detection(img1, img2, 10)
    sim_scores = []
    diff_scores = []
    #after assessing the values of scores with the selected parameters, 350
    #was the acceptable threshold for similar/duplicate photos, i and k correspond to 
    #i and k correspond to the compared photos
    if similarity_scoring[0] <= 350:
        sim_scores.append([similarity_scoring[0],i,k])
    #different scores, in case one wanted to check the different images.
    else:
        diff_scores.append([similarity_scoring[0],i,k])
    # duplicate_images
    # for kl in range(len(sim_scores)):
    #     duplicate_images.append(np.array(sim_scores[kl][2]))
    #     duplicate_images = np.unique(np.array(duplicate_images))
   
    return sim_scores  
#function which takes the duplicate images' number and deletes it from the folder.
def delete_similar_images(duplicate_images):
    for r in range(len(duplicate_images)):
        for u in range(1,441):
            if u == duplicate_images[r]:
               os.remove(path+'\c25/'+str(u)+'.png')
    return duplicate_images       

similarity = []
duplicates = []
duplicate_scoring = []
#change the path when testing the code. 
path = "E:\Kopernikus"
#trial test for the first 10 photos
#looping runs so that the photos can only be compared once, not twice.
array = list(range(0,11))
for i in range(0,len(array)):
    img1 = cv2.imread(path+'\c25/'+str(i)+'.png')
    for k in range(i+1,len(array)):
        img2 = cv2.imread(path+'\c25/'+str(k)+'.png')
        compare = compare_images_and_find_similar(img1,img2)
        similarity.append(compare)
        #similarity list is to save all returned list elements from the called functions
        similarity = [x for x in similarity if x is not None]
#as it returns a list of lists, the command below is to make it only one list.
similarity = [item for sublist in similarity for item in sublist]        
#loop to get the duplicate image numbers to delete it. 
for t in range(len(similarity)):
    duplicates.append(similarity[t][2])
duplicates = np.unique(duplicates)

#calling the above defined function to delete the duplicates. 
delete_similar_images(duplicates)

        
        
 