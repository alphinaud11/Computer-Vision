import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def CalculateIntegral(f):

    s = np.zeros(f.shape, dtype = int)
    ii = np.zeros(f.shape, dtype = int)
    
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):

            s[i,j] = s[i,j-1] + f[i,j] if j != 0 else f[i,j]
            ii[i,j] = ii[i-1,j] + s[i,j] if i != 0 else s[i,j]

    return ii


# p0 and p1 are assumed to be in this form (row, column)
def CalculateLocalSum(ii, p0, p1):

    localSum = ii[p1]
    subtractions = 0

    if (p0[0] != 0):
        localSum -= ii[p0[0]-1, p1[1]]
        subtractions += 1
    
    if (p0[1] != 0):
        localSum -= ii[p1[0], p0[1]-1]
        subtractions += 1
    
    if (subtractions == 2):
        localSum += ii[p0[0]-1, p0[1]-1]
    
    return localSum


# this function makes sure the kernel is contained in the image dimensions
def IsApplicable(reference, pStart, pEnd, imageShape):

    # left border condition
    if (reference[1] + pStart[1] < 0):
        return False
    
    # right border condition
    if (reference[1] + pEnd[1] > imageShape[1]-1):
        return False
    
    # top border condition
    if (reference[0] + pStart[0] < 0):
        return False
    
    # bottom border condition
    if (reference[0] + pEnd[0] > imageShape[0]-1):
        return False

    return True


def DetectEye(integral, n):

    # m => number of rows, n => number of columns 
    m = 0.15 * n

    # p0-14 are transformed to be in this form (row, column)
    p1 = (round(-0.5*m), round(-0.5*n))
    p2 = (0, round(-0.05*n))
    p3 = (0, round(-0.5*n))
    p4 = (round(0.5*m), round(-0.05*n))
    p5 = (round(-0.5*m), round(0.05*n))
    p6 = (0, round(0.5*n))
    p7 = (0, round(0.05*n))
    p8 = (round(0.5*m), round(0.5*n))
    p9 = (round(0.833*m), round(-0.325*n))
    p10 = (round(2*m), round(-0.225*n))
    p11 = (round(0.833*m), round(-0.1*n))
    p12 = (round(2*m), round(0.1*n))
    p13 = (round(0.833*m), round(0.225*n))
    p14 = (round(2*m), round(0.325*n))

    maxScore = float('-inf')
    maxScorePosition = (-1,-1)

    for i in range(integral.shape[0]):
        for j in range(integral.shape[1]):

            if (IsApplicable((i,j), p1, (p14[0], p8[1]), integral.shape)):

                ls1 = CalculateLocalSum(integral, (i+p1[0], j+p1[1]), (i+p2[0], j+p2[1]))
                ls2 = CalculateLocalSum(integral, (i+p3[0], j+p3[1]), (i+p4[0], j+p4[1]))
                ls3 = CalculateLocalSum(integral, (i+p5[0], j+p5[1]), (i+p6[0], j+p6[1]))
                ls4 = CalculateLocalSum(integral, (i+p7[0], j+p7[1]), (i+p8[0], j+p8[1]))
                ls5 = CalculateLocalSum(integral, (i+p9[0], j+p9[1]), (i+p10[0], j+p10[1]))
                ls6 = CalculateLocalSum(integral, (i+p11[0], j+p11[1]), (i+p12[0], j+p12[1]))
                ls7 = CalculateLocalSum(integral, (i+p13[0], j+p13[1]), (i+p14[0], j+p14[1]))

                score = ls1 - ls2 + ls3 - ls4 - ls5 + ls6 - ls7

                if (score > maxScore):
                    maxScore = score
                    maxScorePosition = (i,j)
    
    return maxScorePosition


def ExtractDetectedEye(image, maxPosition, n):

    outputImage = np.zeros(image.shape, dtype = int)
    m = 0.15 * n

    rowStart = maxPosition[0] - round(0.5*m)
    rowEnd = maxPosition[0] + round(2*m)
    colStart = maxPosition[1] - round(0.5*n)
    colEnd = maxPosition[1] + round(0.5*n)

    outputImage[rowStart:rowEnd, colStart:colEnd] = image[rowStart:rowEnd, colStart:colEnd]

    return outputImage


if __name__ == "__main__":

    # reading the image
    image = Image.open("f3.jpg")
    # converting the image to grayscale
    image = image.convert('L')
    # converting the image to 2D array
    image = np.array(image)
    
    n = 250
    integral =  CalculateIntegral(image)
    maxScorePosition = DetectEye(integral, n)
    outputImage = ExtractDetectedEye(image, maxScorePosition, n)

    plt.imshow(outputImage, cmap='gray')
    plt.show()