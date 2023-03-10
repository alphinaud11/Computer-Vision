import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def CalculateColorMap(image):

    width = image.size[0]
    height = image.size[1]

    # height => number of rows, width => number of columns
    index_map = np.zeros([height, width], dtype = int)
    color_map = {}
    index = 1

    for i in range(0, width):
        for j in range(0, height):

            r, g, b = image.getpixel((i,j))

            # if pixel color is not in color map, then we put it
            if (not ((r, g, b) in color_map)):
                color_map[(r, g, b)] = index
                index += 1

            # filling the pixel location in the index map with the pixel color's index
            index_map[j][i] = color_map[(r, g, b)]

    return index_map, color_map


def QuantizationLevels(color_map, range):

    new_color_map = {}
    removed_colors = {}

    for (r, g, b), index in color_map.items():

        match = False
        match_index = -1

        for (r1, g1, b1), index1 in new_color_map.items():
            r_diff = abs(r - r1)
            g_diff = abs(g - g1)
            b_diff = abs(b - b1)
            if (r_diff <= range and g_diff <= range and b_diff <= range):
                match = True
                match_index = index1
                break
        
        if (match):
            removed_colors[index] = match_index
        else:
            new_color_map[(r, g, b)] = index

    return new_color_map, removed_colors


def AdjustIndex(index_map, removed_colors):

    adjusted_index_map = np.zeros(index_map.shape, dtype = int)

    for i in range(0, index_map.shape[0]):
        for j in range(0, index_map.shape[1]):

            if (index_map[i][j] in removed_colors):
                adjusted_index_map[i][j] = removed_colors[index_map[i][j]]
            else:
                adjusted_index_map[i][j] = index_map[i][j]

    return adjusted_index_map


def ReverseColorMap(new_color_map):

    reversed_color_map = {}

    for (r, g, b), index in new_color_map.items():
        reversed_color_map[index] = (r, g, b)

    return reversed_color_map    


def ColorMapToImage(index_map, color_map):

    width = index_map.shape[1]
    height = index_map.shape[0]

    image = Image.new("RGB", (width, height))

    for i in range(0, width):
        for j in range(0, height):

            rgb = color_map[index_map[j][i]]
            image.putpixel((i, j), rgb)

    return image


if __name__ == "__main__":

    image = Image.open("Test Image.jpg")

    index_map, color_map = CalculateColorMap(image)

    print(len(color_map))

    new_color_map, removed_colors = QuantizationLevels(color_map, 30)

    print(len(new_color_map))

    adjusted_index_map = AdjustIndex(index_map, removed_colors)

    reversed_color_map = ReverseColorMap(new_color_map)

    quantized_image = ColorMapToImage(adjusted_index_map, reversed_color_map)

    plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
    
    #original image reconstructed from indexed image
    imgR = ColorMapToImage(index_map, ReverseColorMap(color_map))
    plt.subplot(2,2,1)
    plt.imshow(imgR)

    #indexed image of the original image
    plt.subplot(2,2,2)
    plt.imshow(index_map, cmap = 'gray')

    #image reconstructed after quantization by certain range
    plt.subplot(2,2,3)
    plt.imshow(quantized_image)

    #indexed image of the quantized image by certain range
    plt.subplot(2,2,4)
    plt.imshow(adjusted_index_map, cmap = 'gray')

    plt.show()