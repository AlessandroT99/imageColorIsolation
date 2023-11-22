#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow, imread
from skimage.color import rgb2hsv, hsv2rgb
import cv2
from time import sleep
from scipy.ndimage import label

# FUNCTION AND DEFs ---------------------------------------------------------------------------------------------------------------------------------
def filteringImage(image_path):
    """
    filteringImage()
    -------------------
    The image is analyzed inside this function in order to output an image with only   \
        the yellow pointer as white squares and a matrix in 2D representing the mask of the image.
    
    ### INPUTS
    * `image_path`: path of the image that you want to analyze.
    ### OUTPUTS
    * `dataMatrix`: contains the mask of the image in 2D.
    """
    img = cv2.imread(image_path)
    # Plot the original image
    printOriginalImage(image_path)
    
    # Define the extracted image in hsv
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Plot results
    #hsv_list = ['Hue','Saturation','Value']
    #fig, ax = plt.subplots(1, 3, figsize=(15,7), sharey = True)
    
    #pcm = ax[0].imshow(img_hsv[:,:,0], cmap = 'hsv')
    #ax[0].set_title(hsv_list[0], fontsize = 20)
    #fig.colorbar(pcm, ax=ax[0])
    #ax[0].axis('off')
    
    #pcm = ax[1].imshow(img_hsv[:,:,1], cmap = 'Greys')
    #ax[1].set_title(hsv_list[1], fontsize = 20)
    #fig.colorbar(pcm, ax=ax[1])
    #ax[1].axis('off')
    
    #pcm = ax[2].imshow(img_hsv[:,:,2], cmap = 'gray')
    #ax[2].set_title(hsv_list[2], fontsize = 20)
    #fig.colorbar(pcm, ax=ax[2])
    #ax[2].axis('off')
    
    #fig.tight_layout()

    # Found values of the pointer between 24 and 28 for the hsv and over 80 for saturation
    # Mask creation
    hue_lower_mask = img_hsv[:,:,0] < 28
    hue_upper_mask = img_hsv[:,:,0] > 20
    saturation_mask = img_hsv[:,:,1] > 95 
    mask = hue_lower_mask*hue_upper_mask*saturation_mask
    # [367-377, 261-274] are the [x,y] values where the soap pointer is
    # [370-397, 294-320] are the [x,y] values to obscure in order to not find the wrong soap point 
    mask[294:320,370:397] = 0
    mask[261:274,367:377] = 1
    # mask the part not analyzed 
    mask[0:200,:] = 0
    mask[:,0:100] = 0
    # Mask application
    red = img[:,:,0]*mask
    green = img[:,:,1]*mask
    blue = img[:,:,2]*mask
    img_masked = np.dstack((red,green,blue))
    for i in range(len(img_masked[:,0,0])):
    	for j in range(len(img_masked[0,:,0])):
    	    for k in range(len(img_masked[0,0,:])):
                if img_masked[i,j,k] > 0:
                    img_masked[i,j,k] = 255
                else:
                    img_masked[i,j,k] = 0
    # Show results
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(img_masked);
    
    # Usefull matrix
    dataMatrix = img_masked[:,:,0]/255
    return dataMatrix
    
def printOriginalImage(image_path):
    """
    printOriginalImage()
    -------------------
    Its only function is to print the original image before analyzing it.
    
    ### INPUTS
    * `image_path`: path of the image that you want to analyze.
    ### OUTPUTS
    * None.
    """
    image = imread(image_path)
    plt.figure(num=None,figsize=(8,6),dpi=80)
    plt.imshow(image)

def find_polygon_centers(matrix):
    """
    find_polygon_centers()
    -------------------
    The three pointers represented as polygons are analyzed to get their centers
    
    ### INPUTS
    * `matrix`: contains the mask of the image in 2D.
    ### OUTPUTS
    * `polygons`: contains the polygon centers 
    """
    labeled_matrix, num_labels = label(matrix)
    
    polygons = []
    for i in range(1, num_labels + 1):
        polygon = np.argwhere(labeled_matrix == i)
        center = np.mean(polygon, axis=0, dtype=int)
        polygons.append(center)
    
    return polygons

def distance_between_two_points(point1, point2):
    """
    distance_between_two_points()
    -------------------
    Evaluate the distance between two points.
    
    ### INPUTS
    * `point1`: Right point of them
    ### OUTPUTS
    * `point2`: Left point of them
    """
    return np.linalg.norm(point1 - point2)

def wireMeasures(matrix):
    """
    wireMeasures()
    -------------------
    Evaluate the distance and the angle between the lateral point and the central of the wire 
    
    ### INPUTS
    * `matrix`: path of the image that you want to analyze.
    ### OUTPUTS
    * `leftLength`: The lenght of the left part of the wire (looking from the frontside)
    * `rightLength`: The lenght of the right part of the wire (looking from the frontside)
    * `leftAngle`: The angle of the left part of the wire from the central reference (looking from the frontside)
    * `rightAngle`: The angle of the right part of the wire from the central reference (looking from the frontside)
    """
    # Find polygon centers
    polygon_centers = find_polygon_centers(matrix)

    # Assume there are three polygons
    if len(polygon_centers) == 3:
        center1, center2, center3 = polygon_centers

        # Evaluate distance between the two polygons at the sides from the one in the middle
        leftLength = distance_between_two_points(center1, center2)
        rightLength = distance_between_two_points(center3, center2)

        print("Distance between the two polygons at the sides from the one in the middle:")
        print("- Left side:  ", leftLength)
        print("- Right side: ", rightLength)
    else:
        raise(ValueError,"The number of found pointer in the image is not equal to 3.")

# VARIABLES AND CONSTANTS ---------------------------------------------------------------------------------------------------------------------------
numImages = 7

# MAIN PROGRAM --------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    for i in range(1,numImages):
        path = f"trial/{str(i)}.ppm"
        try:
            dataMask = filteringImage(path)
            wireMeasures(dataMask)

            plt.show()

        except Exception as err: 
            print(f"In image {str(i)} has been found an error: ", err)
            exit
