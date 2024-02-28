#!/usr/bin/env python3

import os
import csv
import cv2
import math
import telepot 
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow, imread
from scipy.ndimage import label
from typing import List, Type
from operator import itemgetter
from time import sleep

# FUNCTION AND DEFs ---------------------------------------------------------------------------------------------------------------------------------
def filteringImage(image_path, subjectNumber, imageNumber):
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
    if SHOW_PLOT:
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
        plt.show()

    # Found values of the pointer between 20 and 30 for the hsv and over 80 for saturation
    # Mask creation
    hue_lower_mask = img_hsv[:,:,0] < 30
    hue_upper_mask = img_hsv[:,:,0] > 20
    saturation_mask = img_hsv[:,:,1] > 95
    # Create a mask for the specified color range
    mask = hue_lower_mask*saturation_mask*hue_upper_mask

    # Remove the clamp yellow
    if subjectNumber < 8:
        mask[600:670,700:770] = 0
    elif subjectNumber < 22:
        mask[580:690,550:670] = 0
    else:
        mask[580:700,630:730] = 0

    mask[soapLimits[subjectNumber,0]:soapLimits[subjectNumber,1],soapLimits[subjectNumber,2]:soapLimits[subjectNumber,3]] = 0
    mask[soapLimits[subjectNumber,4]:soapLimits[subjectNumber,5],soapLimits[subjectNumber,6]:soapLimits[subjectNumber,7]] = 1

    # Show results
    if SHOW_PLOT or imageNumber == 0:
        # Apply the mask to the original image
        reverseMask = -(mask-1)
        red = img[:,:,0]*reverseMask
        green = img[:,:,1]*reverseMask
        blue = img[:,:,2]*reverseMask
        img_masked = np.dstack((red,green,blue))
        plt.figure(num=None, figsize=(8, 6), dpi=80)
        plt.imshow(img_masked)
        plt.show(block=False)
        plt.pause(15)
        plt.close()
    
    # Usefull matrix
    return mask
    
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
    global collectedErrors, bot

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

        if DEBUG:
            print("Distance between the two polygons at the sides from the one in the middle:")
            print(f"- Left side:  {str(round(leftLength,2))} cm")
            print(f"- Right side: {str(round(rightLength,2))} cm")
            print(f"Total Length of the wire: {str(round(leftLength+rightLength,2))} cm")

        # Evaluate angles
        xL = pixel2cm(centerPointer[1],"x")-pixel2cm(leftPointer[1],"x")
        leftAngle = math.acos(xL/leftLength)*180/math.pi
        xR = pixel2cm(rightPointer[1],"x")-pixel2cm(centerPointer[1],"x")
        rightAngle = math.acos(xR/rightLength)*180/math.pi

        if DEBUG:
            print("\nAngles of the cables between the horizontal soap plane:")
            print(f"- Left side:  {str(round(leftAngle,2))} deg")
            print(f"- Right side: {str(round(rightAngle,2))} deg")
            sleep(0.5)
    else:
        if collectedErrors < ADMITTED_ERRORS:
            collectedErrors += 1
            leftLength = -1 
            rightLength = -1 
            leftAngle = -1 
            rightAngle = -1
        else: 
            raise ValueError("The number of found pointer in the image is not equal to 3.")
    
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
IMAGE_WIDTH_PX = 1280   # Width of the picture analyzed (equal for all of them)
IMAGE_HEIGTH_PX = 720   # Height of the picture analyzed (equal for all of them)
IMAGE_PPI = 144         # PPI resolution of the picture analyzed

WIRE_REAL_LENGTH = 800  # [mm] is the length of the cable in the real world (considering the end of the yellow marker)
WIRE_IMAGE_LENGTH = 80  # [mm] evaluated from test analysis

SHOW_PLOT = 0           # 0 if no showing images, 1 if showing all images and process
DEBUG = 0               # 0 if normal functioning, 1 if entering in debug mode and display more details of whats happening during the execution
ADMITTED_ERRORS = 50    # Number of admitted error in the polygons evaluation

NOT_COUNTED_FILES = 3   # Number of files in the experiment folder to not count (info, data and video files)

TELEGRAM_LOG = 1        # 0 if the log is not reported

soapLimits = np.array([[0,720,0,200,370,385,750,768],       #1R
                       [0,720,0,200,370,385,750,768],       #2R
                       [0,720,0,200,370,385,750,768],       #3R
                       [0,720,0,200,370,385,750,768],       #4R
                       [0,720,0,200,370,385,750,768],       #5R
                       [0,720,0,200,370,385,750,768],       #6R
                       [0,720,0,200,370,385,750,768],       #7R
                       [0,720,0,170,375,390,580,592],       #8L
                       [0,720,0,170,375,390,580,592],       #9L
                       [0,720,0,170,375,390,580,592],       #10L
                       [0,720,0,170,375,390,580,592],       #11L
                       [0,720,0,170,375,390,580,592],       #12L
                       [0,720,0,170,375,390,580,592],       #13L
                       [0,720,0,170,375,390,580,592],       #14L
                       [0,720,0,170,375,390,580,592],       #15L
                       [0,0,0,0,0,0,0,0],                   #16L
                       [0,720,0,170,375,390,580,592],       #17L
                       [0,720,0,170,375,390,580,592],       #18L
                       [0,720,0,170,375,390,580,592],       #19L
                       [0,720,0,170,375,390,580,592],       #20L
                       [0,720,0,170,375,390,580,592],       #21L
                       [0,720,0,180,372,390,692,708],       #22R
                       [0,720,0,180,372,390,692,708],       #23R
                       [0,720,0,180,372,390,670,690],       #24R
                       [0,720,0,180,372,390,692,708],       #25R
                       [0,720,0,180,372,390,680,695],       #26R
                       [0,720,0,180,372,390,692,708],       #27R
                       [0,720,0,180,372,390,692,708],       #28R
                       [0,720,0,180,372,390,692,708],       #29R
                       [0,0,0,0,0,0,0,0]])                  #30L

# Telegram Bot creation for notification
userName = 486322403
TOKEN = "5359903695:AAEVr-HqTxjIMSFIChgmwq-9RiWuMuiwFtQ"
bot = telepot.Bot(TOKEN)

# MAIN PROGRAM --------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n")
    titleName = " HRI soap cutting experiment - Image processing "
    print(titleName.center(80,"-"))
    if TELEGRAM_LOG:
        bot.sendMessage(userName, "Image processing for cutting experiment started")

    # Initializations
    numTests = 0
    dir_path = "../images" # Directory where the data of all test are placed

    # If does not exist reate folder for sub-results of each experiment
    if not os.path.exists("SingleExperimentData"): 
       os.makedirs("SingleExperimentData") 

    # Checking folder number
    for path in os.listdir(dir_path):
        # Check if current path is a directory
        if os.path.isdir(os.path.join(dir_path, path)):
            numTests += 1
    numTests -= 1 #removing the folder P1_00001
    errorInput = 1
    while errorInput == 1:
        errorInput = 0
        firstTest = int(input(f"\nThere are {str(numTests)} available, from which do you want to start? [1:{str(numTests)}]: "))
        if firstTest < 1 or firstTest > numTests:
            errorInput = 1
            print(f"\nNope, you ask something impossible, retry.\n")
        else:
            lastTest = int(input(f"Ok, starting from {str(firstTest)}, and at which do you want to end? [{str(firstTest)}:{str(numTests)}]: "))
            if lastTest < firstTest or lastTest > numTests:
                errorInput = 1
                print(f"\nNope, you ask something impossible, retry.\n")
        
    # Do the same procedure for all the folders and so for all the tests
    # Output file creation
    with open(f"ImageProcessingOutput_{str(firstTest)}-{str(lastTest)}",'w',newline='') as outsideFile:
        outsideWriter = csv.writer(outsideFile)
        outsideWriter.writerow(["ID","leftLength_mean", "leftLength_std", "rightLength_mean", "rightLength_std", "totalLength_mean", "totalLength_std" \
                                "leftAngle_mean", "leftAngle_std", "rightAngle_mean", "rightAngle_std"])
        try:
            # Elaborate the results for each folder
            cnt = firstTest
            while cnt < lastTest+1:
                # Avoid test with no data or discarded
                if cnt in [16, 30]:
                    toPrint = f"[DISCARD] Test {str(cnt)} discarded."
                    print("\n" + toPrint)
                    if TELEGRAM_LOG:
                        bot.sendMessage(userName, toPrint)
                    cnt += 1
                    if cnt > numTests:
                        break

                toPrint = f"[START] Starting processing on test {str(cnt)}..."
                print("\n" + toPrint)
                if TELEGRAM_LOG:
                    bot.sendMessage(userName, toPrint)
                numImages = filesCounter(cnt)

                leftLengthArray = [] 
                rightLengthArray = [] 
                leftAngleArray = [] 
                rightAngleArray = []
                totalLengthArray = []
                collectedErrors = 0
                # Create single output file
                with open(f"SingleExperimentData/Test{str(cnt)}_ImageProcessingData",'w',newline='') as insideFile:
                    insideWriter = csv.writer(insideFile)
                    insideWriter.writerow(["leftLength", "rightLength", "totalLength", "leftAngle", "rightAngle"])
                    i = 0
                    while i < numImages:
                        try:
                            path = f"../images/P_{str(cnt).rjust(5,'0')}/{str(i).rjust(8,'0')}.ppm"
                            dataMask = filteringImage(path,cnt-1,i)

                            leftLength, rightLength, leftAngle, rightAngle = wireMeasures(dataMask)

                            totalLength = leftLength + rightLength
                            # An error is been found but is still admitted so its just reported
                            if leftLength == -1 and rightLength == -1 and leftAngle == -1 and rightAngle == -1:
                                if 0: # Avoid printing all these informations
                                    print("\n[WARNING] An error on marker recognising has been found, but is still admissibile.")
                                    if ADMITTED_ERRORS-collectedErrors > 0:
                                        print(f"          Only {str(ADMITTED_ERRORS-collectedErrors)} admissible errors remaining.")
                                    else: 
                                        print(f"          No more admissible errors remaining.")
                            else:
                                # Save data for the average at the end
                                leftLengthArray.append(leftLength)
                                rightLengthArray.append(rightLength)
                                leftAngleArray.append(leftAngle)
                                rightAngleArray.append(rightAngle)
                                totalLengthArray.append(totalLength)
                                # Save data into the single folder output file
                                insideWriter.writerow([leftLength, rightLength, totalLength, leftAngle, rightAngle])

                            if i == round(numImages/4):
                                text = "[INFO] Test processing status: 25" + '%' + " completed."
                                print("\n" + text)
                                #if TELEGRAM_LOG:
                                #    bot.sendMessage(userName, text)
                            elif i == round(numImages/2):
                                text = "[INFO] Test processing status: 50" + '%' + " completed."
                                print("\n" + text)
                                if TELEGRAM_LOG:
                                    bot.sendMessage(userName, text)
                            elif i == round(numImages*3/4):
                                text = "[INFO] Test processing status: 75" + '%' + " completed."
                                print("\n" + text)
                                #if TELEGRAM_LOG:
                                #    bot.sendMessage(userName, text)
                            
                            i += 1
                    
                        except ValueError:
                            toPrint = f"[ERROR] Error found, limit of {str(ADMITTED_ERRORS)} admissible errors overpassed. \nStopping execution for the current dataset (N. {str(cnt)})."
                            print("\n" + toPrint)
                            if TELEGRAM_LOG:
                                bot.sendMessage(userName, toPrint)
                                break
                
                # Save mean data of the just processed test                     
                outsideWriter.writerow([cnt,np.mean(leftLengthArray,dtype=np.float64),np.std(leftLengthArray,dtype=np.float64),
                                        np.mean(rightLengthArray,dtype=np.float64),np.std(rightLengthArray,dtype=np.float64),
                                        np.mean(totalLengthArray,dtype=np.float64),np.std(totalLengthArray,dtype=np.float64),
                                        np.mean(leftAngleArray,dtype=np.float64),np.std(leftAngleArray,dtype=np.float64),
                                        np.mean(rightAngleArray,dtype=np.float64),np.std(rightAngleArray,dtype=np.float64)])

                toPrint = f"[INFO] Dataset {str(cnt)} has a found those mean angle [" + "{:.2f}".format(np.mean(leftAngleArray,dtype=np.float64)) + ", " + "{:.2f}".format(np.mean(rightAngleArray,dtype=np.float64)) +  "]."
                if TELEGRAM_LOG:
                    bot.sendMessage(userName, toPrint)
                print("\n" + toPrint)

                toPrint = f"[END] Processing of dataset {str(cnt)} is done. Has been found {str(collectedErrors)} admissible errors."
                if TELEGRAM_LOG:
                    bot.sendMessage(userName, toPrint)
                print("\n" + toPrint)

                if i == round(numTests/4):
                    text = "[INFO] Analysis processing status: 25" + '%' + " completed."
                    print("\n" + text)
                    if TELEGRAM_LOG:
                        bot.sendMessage(userName, text)
                elif i == round(numTests/2):
                    text = "\n[INFO] Analysis processing status: 50" + '%' + " completed."
                    print("\n" + text)
                    if TELEGRAM_LOG:
                        bot.sendMessage(userName, text)
                elif i == round(numTests*3/4):
                    text = "\n[INFO] Analysis processing status: 75" + '%' + " completed." 
                    print("\n" + text)
                    if TELEGRAM_LOG:
                        bot.sendMessage(userName, text)

                cnt += 1

        except:
            toPrint = "[FATAL ERROR] An error has been founded, stopping execution. Waiting for correction."
            print("\n" + toPrint)
            if TELEGRAM_LOG:
                bot.sendMessage(userName, toPrint)
            exit()

        print("\n[INFO] Image processing over.\n")
        titleName = " by Alessandro Tiozzo - alessandro.tiozzo@iit.it "
        print(titleName.center(80,"-"))
        print("\n")
        if TELEGRAM_LOG:
            bot.sendMessage(userName, "[INFO] Execution completed! Come to see results")
