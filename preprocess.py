import numpy as np
import cv2
import csv
import pandas as pd

#Median subtraction
#Does also affect black area around edges
#https://www.kaggle.com/joorarkesteijn/fast-cropping-preprocessing-and-augmentation
def subtract_median(img):
	k=np.max(img.shape)//20*2+1
	bg=cv2.medianBlur(img,k)
	return cv2.addWeighted(img,4,bg,-4,128)

def crop_image(img, resize_width=299, resize_height=299):
    #Convert to black and gray and threshold
    output = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    #RETR_EXTERNAL finds only extreme outer contours
    #CHAIN_APPROX_SIMPLE compresses segments leaving only the end points
    gray,contours,hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #this catches any images that are too dark. I wasn't able to find any examples to test this though
    if not contours:
        print('No contours! Image is too dark')
        flag = 0
        return img, flag
    #find the largest contour
    cnt = max(contours, key = cv2.contourArea)

    #Get center of circle and radius
    ((x, y), r) = cv2.minEnclosingCircle(cnt)
    x = int(x)
    y = int(y)
    r = int(r)

    #Get height and width of original image and divide by 2
    height = int(np.size(img, 0)/2)
    width = int(np.size(img, 1)/2)

    #if the circle is bigger than the image, return resized original. else crop and then resize
    dim = (resize_width,resize_height)
    r=int(r * 0.8)
    if(r > width and r > height):
        return cv2.resize(output, dim, interpolation=cv2.INTER_AREA)

    else:
        if(r > height):
            output = output[:,max(x-r,0):x+r]
        elif(r > width):
            output = output[max(y-r,0):y+r,:]
        else:
            output = output[max(y-r,0):y+r,max(x-r,0):x+r]
        return cv2.resize(output, dim, interpolation=cv2.INTER_AREA)
        
        
#read in image, do median subtraction, crop and resize, save preprocessed image
#REMEMBER TO CHANGE BACK TO allCells.csv (w/o 2) IN FOLLOWING LINE
df = pd.read_csv('allCells2.csv', usecols = ['image_id', 'Path_to_image'])
#count = 0
for i,j in zip(df['Path_to_image'], df['image_id']):
    img = cv2.imread(i)
    img = subtract_median(img)
    img = crop_image(img)
    cv2.imwrite('data/preprocessed/' + j + '.png', img)
