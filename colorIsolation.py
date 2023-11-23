#!/usr/bin/env python3

from array import array
from re import T
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow, imread
from skimage.color import rgb2hsv, hsv2rgb
import cv2
from time import sleep
from scipy.ndimage import label
from typing import List, Type
import math
from operator import itemgetter
import os

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
    if SHOW_PLOT:   
        hsv_list = ['Hue','Saturation','Value']
        fig, ax = plt.subplots(1, 3, figsize=(15,7), sharey = True)
        
        pcm = ax[0].imshow(img_hsv[:,:,0], cmap = 'hsv')
        ax[0].set_title(hsv_list[0], fontsize = 20)
        fig.colorbar(pcm, ax=ax[0])
        ax[0].axis('off')
        
        pcm = ax[1].imshow(img_hsv[:,:,1], cmap = 'Greys')
        ax[1].set_title(hsv_list[1], fontsize = 20)
        fig.colorbar(pcm, ax=ax[1])
        ax[1].axis('off')
        
        pcm = ax[2].imshow(img_hsv[:,:,2], cmap = 'gray')
        ax[2].set_title(hsv_list[2], fontsize = 20)
        fig.colorbar(pcm, ax=ax[2])
        ax[2].axis('off')
        
        fig.tight_layout()

    # Found values of the pointer between 20 and 30 for the hsv and over 80 for saturation
    # Mask creation
    hue_lower_mask = img_hsv[:,:,0] < 30
    hue_upper_mask = img_hsv[:,:,0] > 20
    saturation_mask = img_hsv[:,:,1] > 95 
    mask = hue_lower_mask*hue_upper_mask*saturation_mask
    # [367-377, 261-274] are the [x,y] values where the soap pointer is
    # [370-397, 294-320] are the [x,y] values to obscure in order to not find the wrong soap point 
    mask[294:320,370:397] = 0
    mask[261:274,367:377] = 1
    # Mask the part not analyzed 
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
    if SHOW_PLOT:
        plt.figure(num=None, figsize=(8, 6), dpi=80)
        plt.imshow(img_masked)
        plt.show()
    
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
    The three pointers represented as polygons are analyzed to get their centers.
    
    ### INPUTS
    * `matrix`: contains the mask of the image in 2D.
    ### OUTPUTS
    * `polygons`: contains the polygon centers sorted by x position.
    """
    labeled_matrix, num_labels = label(matrix)
    
    lengths = []
    for i in range(1, num_labels + 1):
        lengths.append(len(np.argwhere(labeled_matrix == i)))
    sortedLengths = sorted(lengths, reverse=True)

    # Now check only the three bigger polygons
    polygons = []
    for j in range(1, num_labels + 1):
        if lengths[j-1] in sortedLengths[:3]:
            polygon = np.argwhere(labeled_matrix == j)
            center = np.mean(polygon, axis=0, dtype=int)
            polygons.append(center)

    return sorted(polygons, key=itemgetter(1))

def distance_between_two_points(point1: List[float], point2: List[float]) -> float:
    """
    distance_between_two_points()
    -------------------
    Evaluate the distance between two points in norm 1.
    
    ### INPUTS
    * `point1`: Right point of them.
    * `point2`: Left point of them.
    ### OUTPUTS
    * `distance`: Distance between the two points.
    """
    if not type(point1) is list or len(point1) != 2 or not type(point1[0]) is float:
        raise TypeError("The variable point1 inserted in distance_between_two_points() funct is not a length 2 List[float] type")
    elif not type(point2) is list or len(point2) != 2 or not type(point2[0]) is float:
        raise TypeError("The variable point2 inserted in distance_between_two_points() funct is not a length 2 List[float] type")

    distance_xy = [a-b for a,b in zip(point1,point2)]
    distance = np.linalg.norm(distance_xy,2)
    return distance

def wireMeasures(matrix):
    """
    wireMeasures()
    -------------------
    Evaluate the distance and the angle between the lateral point and the central of the wire. 
    
    ### INPUTS
    * `matrix`: Input matrix containing the 2D localization of the pointers.
    ### OUTPUTS
    * `leftLength`: The lenght of the left part of the wire (looking from the frontside).
    * `rightLength`: The lenght of the right part of the wire (looking from the frontside).
    * `leftAngle`: The angle of the left part of the wire from the central reference (looking from the frontside).
    * `rightAngle`: The angle of the right part of the wire from the central reference (looking from the frontside).
    """
    # Find polygon centers
    polygon_centers = find_polygon_centers(matrix)

    # Assume there are three polygons
    if len(polygon_centers) == 3:
        leftPointer, centerPointer, rightPointer = polygon_centers
        #print("Pointer left: ",leftPointer)
        #print("Pointer right: ",rightPointer)
        #print("Pointer center: ",centerPointer)

        # Convert pixels into cm before evaluating distances
        # Evaluate distance between the two polygons at the sides from the one in the middle
        leftLength = distance_between_two_points(pixel2cm(centerPointer), pixel2cm(leftPointer))
        rightLength = distance_between_two_points(pixel2cm(rightPointer), pixel2cm(centerPointer))

        print("Distance between the two polygons at the sides from the one in the middle:")
        print(f"- Left side:  {str(round(leftLength,2))} cm")
        print(f"- Right side: {str(round(rightLength,2))} cm")
        print(f"Total Length of the wire: {str(round(leftLength+rightLength,2))} cm")

        # Evaluate angles
        xL = pixel2cm(centerPointer[1],"x")-pixel2cm(leftPointer[1],"x")
        leftAngle = math.acos(xL/leftLength)*180/math.pi
        xR = pixel2cm(rightPointer[1],"x")-pixel2cm(centerPointer[1],"x")
        rightAngle = math.acos(xR/rightLength)*180/math.pi

        print("\nAngles of the cables between the horizontal soap plane:")
        print(f"- Left side:  {str(round(leftAngle,2))} deg")
        print(f"- Right side: {str(round(rightAngle,2))} deg")
    else:
        raise(ValueError,"The number of found pointer in the image is not equal to 3.")
    
    return leftLength, rightLength, leftAngle, rightAngle
    
def pixel2cm(px: List[int], axis: str = ""):
    """
    pixel2cm()
    -------------------
    Tranform (following the image dimension setted in the constants) pixels in cm. 
    
    ### INPUTS
    * `px`: Measure in px of [y,x].
    * `axis`: Contains the axis name if the input is a singular value and not a list of int.
    ### OUTPUTS
    * `cm`: Measure in cm of [y,x].
    """
    height_cm = IMAGE_HEIGTH_PX/IMAGE_PPI
    width_cm = IMAGE_WIDTH_PX/IMAGE_PPI
    if axis == "":
        if not type(px) is np.ndarray or len(px) > 2 or not type(px[0]) is np.int64:
            raise TypeError("The variable px inserted in pixel2cm() funct is not a length 2 list[int] type")
        cm = []
        cm.append(float(px[0])/IMAGE_HEIGTH_PX*height_cm)
        cm.append(float(px[1])/IMAGE_WIDTH_PX*width_cm)
    elif axis.lower() == "x":
        cm = float(px)/IMAGE_WIDTH_PX*width_cm
    elif axis.lower() == "y":
        cm = float(px)/IMAGE_HEIGTH_PX*height_cm
    else:
        raise TypeError("The character inserted for axis variable as not being recognised in pixel2cm()")
    
    #print(cm)
    return cm

def filesCounter(folderNumber: int) -> int:
    """
    filesCounter()
    -------------------
    Counts the number of files into the images folder which number is passed in input. 
    
    ### INPUTS
    * `folderNumber`: Number of the images folder to consider.
    ### OUTPUTS
    * `count`: Number of files inside of the folder.
    """
    if not type(folderNumber) is int:
        raise TypeError("The number of the folder in folderCounter() has to be an integer")

    # Folder path
    dir_path = "../images/P_" + str(folderNumber).rjust(5,"0")
    count = 0
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    #print('File count:', count)
    return count-NOT_COUNTED_FILES

# VARIABLES AND CONSTANTS ---------------------------------------------------------------------------------------------------------------------------
IMAGE_WIDTH_PX = 640    # Width of the picture analyzed (equal for all of them)
IMAGE_HEIGTH_PX = 480   # Height of the picture analyzed (equal for all of them)
IMAGE_PPI = 72          # PPI resolution of the picture analyzed

SHOW_PLOT = 0           # 0 if no showing images, 1 if showing all images and process

NOT_COUNTED_FILES = 3   # Number of files in the experiment folder to not count (info, data and video files)

# MAIN PROGRAM --------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    numTests = 0
    dir_path = "../images"
    for path in os.listdir(dir_path):
        # Check if current path is a directory
        if os.path.isdir(os.path.join(dir_path, path)):
            numTests += 1
    # Do the same procedure for all the folders and so for all the tests
    numTests = 2

    for k in range(1,numTests+1):
        #numImages = filesCounter(k)
        numImages = 10

        leftLengthArray = [] 
        rightLengthArray = [] 
        leftAngleArray = [] 
        rightAngleArray = []

        for i in range(0,numImages):
            path = f"../images/P_{str(k).rjust(5,'0')}/{str(i).rjust(8,'0')}.ppm"
            dataMask = filteringImage(path)

            leftLength, rightLength, leftAngle, rightAngle = wireMeasures(dataMask)
            leftLengthArray.append(leftLength)
            rightLengthArray.append(rightLength)
            leftAngleArray.append(leftAngle)
            rightAngleArray.append(rightAngle)