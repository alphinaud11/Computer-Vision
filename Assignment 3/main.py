import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from numpy.lib import histogram
from numpy.lib.function_base import average


def CalculateHistogram(image):

    imageArr = np.array(image)
    histogram = [0]*256

    imageArr = imageArr.flatten()

    for i in range(imageArr.size):
        histogram[imageArr[i]] += 1
    
    return histogram


def CalculateCumulativeHistogram(histogram):

    cumulativeHistogram = [0]*256

    cumulativeHistogram[0] = histogram[0]

    for i in range(1, 256):
        cumulativeHistogram[i] = histogram[i] + cumulativeHistogram[i-1]

    return cumulativeHistogram


def CalculateEqualizedHistogram(cumulativeHistogram):

    equalizedHistogram = [0]*256
    size = cumulativeHistogram[255]

    for i in range(256):
        equalizedHistogram[i] = round((255 / (size - cumulativeHistogram[0])) * (cumulativeHistogram[i] - cumulativeHistogram[0]))
    
    return equalizedHistogram
        

def CalculateEqualizedImage(image, equalizedHistogram):

    # height => number of rows, width => number of columns
    width = image.size[0]
    height = image.size[1]

    equalizedImage = Image.new("L", (width, height))

    for i in range(0, width):
        for j in range(0, height):

            oldIntensity = image.getpixel((i,j))
            newIntensity = equalizedHistogram[oldIntensity]
            equalizedImage.putpixel((i,j), newIntensity)
    
    return equalizedImage
    

def SegmentOptimalThresholding(image):

    # height => number of rows, width => number of columns
    width = image.size[0]
    height = image.size[1]

    B = []
    O = []

    # first iteration
    for i in range(0, width):
        for j in range(0, height):
            if ((i == 0 and j == 0) or (i == 0 and j == height-1) or (i == width-1 and j == 0) or (i == width-1 and j == height-1)):
                B += [image.getpixel((i,j))]
            else:
                O += [image.getpixel((i,j))]
    
    meanB = average(B)
    meanO = average(O)
    previousThreshold = -1
    currentThreshold = (meanB + meanO) / 2

    while (True):

        B = []
        O = []

        for i in range(0, width):
            for j in range(0, height):
                if (image.getpixel((i,j)) < currentThreshold):
                    B += [image.getpixel((i,j))]
                else:
                    O += [image.getpixel((i,j))]
        
        meanB = average(B)
        meanO = average(O)
        previousThreshold = currentThreshold
        currentThreshold = (meanB + meanO) / 2
        
        if (previousThreshold == currentThreshold):
            break
    
    segmentedImage = Image.new("L", (width, height))

    for i in range(0, width):
        for j in range(0, height):
            if (image.getpixel((i,j)) < currentThreshold):
                segmentedImage.putpixel((i,j), 0)
            else:
                segmentedImage.putpixel((i,j), 255)
    
    return segmentedImage


if __name__ == "__main__":

    # reading the image
    image = Image.open("5.jpg")
    # converting the image to grayscale
    image = image.convert('L')
    
    histogram = CalculateHistogram(image)
    cumulativeHistogram = CalculateCumulativeHistogram(histogram)
    equalizedHistogram = CalculateEqualizedHistogram(cumulativeHistogram)
    equalizedImage = CalculateEqualizedImage(image, equalizedHistogram)
    segmentedOriginalImage = SegmentOptimalThresholding(image)
    segmentedEqualizedImage = SegmentOptimalThresholding(equalizedImage)
    
    
    plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
    
    #original image
    plt.subplot(2,2,1)
    plt.imshow(image, cmap = 'gray')
    plt.title("Original image")

    #segmented original image
    plt.subplot(2,2,2)
    plt.imshow(segmentedOriginalImage, cmap = 'gray')
    plt.title("Segmenting the original image")

    #equalized image
    plt.subplot(2,2,3)
    plt.imshow(equalizedImage, cmap = 'gray')
    plt.title("Equalized image")

    #segmented equalized image
    plt.subplot(2,2,4)
    plt.imshow(segmentedEqualizedImage, cmap = 'gray')
    plt.title("Segmented equalized image")

    plt.show()