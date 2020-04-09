# Import required libraries
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import sys
import time

current_milli_time = lambda: int(round(time.time() * 1000))


# Read source and mask (if exists) for a given id
def Read(id, path = ""):
    source = readColor(id, path)
    maskPath = path + "mask_" + id + ".jpg"
    
    if os.path.isfile(maskPath):
        mask = cv2.imread(maskPath, cv2.IMREAD_COLOR)
        assert(mask.shape == source.shape), 'size of mask and image does not match'
        mask = (mask > 128)[:, :, 0].astype(int)
    else:
        mask = np.zeros_like(source)[:, :, 0].astype(int)

    return source, mask

def readColor(id, path = ""):
    toReturn = cv2.imread(path + "image_" + id + ".jpg", cv2.IMREAD_COLOR)
    return toReturn

def readGrayscale(id, path = ""):
    toReturn = cv2.imread(path + "image_" + id + ".jpg", cv2.IMREAD_GRAYSCALE)
    return toReturn


def SeamCarve(inputImage, widthFac, heightFac, mask):

    # Main seam carving function. This is done in three main parts: 1)
    # computing the energy function, 2) finding optimal seam, and 3) removing
    # the seam. The three parts are repeated until the desired size is reached.

    assert (widthFac == 1 or heightFac == 1), 'Changing both width and height is not supported!'
    assert (widthFac <= 1 and heightFac <= 1), 'Increasing the size is not supported!'

    divider = widthFac
    if (heightFac != 1):
        inputImage = cv2.rotate(inputImage, cv2.ROTATE_90_CLOCKWISE)
        divider = heightFac

    inSize = inputImage.shape
    size   = (int(inSize[0]), int(divider * inSize[1]), 3)

    toReturn = inputImage
    print (size)

    while (toReturn.shape != size):
        # Step 1: Generate energy function
        energyFunc = generateEnergyFunction(toReturn)        
        energyFunc = energyFunc.astype(np.uint16)

        # print("1. Current Time =", current_milli_time())

        # Step 2: Find the optimal seam
        seamToRemove = findOptimalSeam(energyFunc)
        # print("2 seam. Current Time =", current_milli_time())
        
        # Step 3: Remove the seam
        toReturn = removeSeam(toReturn, seamToRemove)

        # print("3. Current Time =", current_milli_time())

        print("Image Size: " + str(toReturn.shape))

    if (heightFac != 1):
        toReturn = cv2.rotate(toReturn, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return toReturn, toReturn.shape[:2]

# Calculates and returns energy function for given image
def generateEnergyFunction(img):
    # Convert image to grayscale
    grayScaleImage = cv2.cvtColor(img.astype(np.uint16), cv2.COLOR_BGR2GRAY)
    # Get gradients
    sobelX = cv2.convertScaleAbs(cv2.Sobel(grayScaleImage,cv2.CV_64F,1,0,ksize=3))
    sobelY = cv2.convertScaleAbs(cv2.Sobel(grayScaleImage,cv2.CV_64F,0,1,ksize=3))

    # Create energy map
    energyMap = cv2.addWeighted(sobelX, 0.5, sobelY, 0.5, 0)
    return energyMap

# Finds the optimal seam using the energy function
def findOptimalSeam(energyFunc):

    numRows, numCols = energyFunc.shape
    costMatrix = energyFunc.copy()
    seam = np.zeros(numRows)

    # Calculate the cost matrix
    for rowIndex in range(1, numRows):
        for colIndex in range (0, numCols):
            if (colIndex == 0):
                costMatrix [rowIndex] [colIndex] += min([costMatrix[rowIndex - 1][colIndex], costMatrix[rowIndex - 1][colIndex + 1]])
            elif (colIndex == numCols - 1):
                costMatrix [rowIndex] [colIndex] += min([costMatrix[rowIndex - 1][colIndex - 1], costMatrix[rowIndex - 1][colIndex]])
            else:
                costMatrix [rowIndex] [colIndex] += min([costMatrix[rowIndex - 1][colIndex - 1], costMatrix[rowIndex - 1][colIndex], costMatrix[rowIndex - 1][colIndex + 1]])
    
    # Start from the bottom and find the optimal seam
    seam[numRows - 1] = int(np.argmin(costMatrix[numRows - 1]))
    for rowIndex in range(numRows - 2, -1, -1):
        lastSeamIndex = int(seam[rowIndex + 1])
        if (lastSeamIndex == 0):
            seam[rowIndex] = int(np.argmin([costMatrix[rowIndex][lastSeamIndex], costMatrix[rowIndex][lastSeamIndex + 1]]))
        elif (lastSeamIndex == numCols - 1):
            seam[rowIndex] = int(lastSeamIndex - 1 + np.argmin([costMatrix[rowIndex][lastSeamIndex - 1], costMatrix[rowIndex][lastSeamIndex]]))
        else:
            seam[rowIndex] = int(lastSeamIndex + np.argmin([costMatrix[rowIndex][lastSeamIndex - 1], costMatrix[rowIndex][lastSeamIndex], costMatrix[rowIndex][lastSeamIndex + 1]]) - 1)

    return seam



# Remove the seam from the image
def removeSeam(image, seam):
    numRows, numCols, _ = image.shape
    #remove seam from image and adjust all other cols
    for rowIndex in range(0, numRows):
        #remove seam from imae and move over all other cols
        for colIndex in range(int(seam[rowIndex]), numCols - 1):
            image[rowIndex][colIndex] = image[rowIndex] [colIndex + 1]
    small_image = image [:, 0:numCols-1]
    return small_image


# Gets the index of the smallest element
def getMinIndex(aRow):
    aList = list(aRow)
    return aList.index(min(aList))

# Scales image to uint8
def scaleTo8Bit(image, displayMin = None, displayMax = None):
    if displayMin == None:
        displayMin = np.min(image)

    if displayMax == None:
        displayMax = np.max(image)

    np.clip(image, displayMin, displayMax, out = image)

    image = image - displayMin
    cf = 255. / (displayMax - displayMin)
    imageOut = (cf * image).astype(np.uint8)
    return imageOut

# Setting up the input output paths
inputDir = '../Images/'
outputDir = '../Results/'


N = 4 # number of images

for index in range(1, N + 1):

    inputImage, mask = Read(str(index).zfill(2), inputDir)

    # Seam Carve for half the width
    output, size = SeamCarve(inputImage, 0.5, 1, mask)
    # Writing the result
    cv2.imwrite("{}/result_{}_{}x{}.jpg".format(outputDir, 
                                            str(index).zfill(2), 
                                            str(size[1]).zfill(2), 
                                            str(size[0]).zfill(2)), output)
    
    print("Image " + str(index) + " by width done")

    inputImage, mask = Read(str(index).zfill(2), inputDir)

    # Seam Carve for half the height
    output, size = SeamCarve(inputImage, 1, 0.5, mask)
    # Writing the result
    cv2.imwrite("{}/result_{}_{}x{}.jpg".format(outputDir, 
                                            str(index).zfill(2), 
                                            str(size[1]).zfill(2), 
                                            str(size[0]).zfill(2)), output)
    print("Image " + str(index) + " by height done")
