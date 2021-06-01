import sys
import cv2 # pip install opencv-python
import numpy as np # pip install numpy


# Grayscale Image
def processImage(image):
    image = cv2.imread(image)
   # image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    return image


def convolve2D(image, kernel, padding=0, strides=1, out_channels=5):
    # Cross Correlation
   # kernel = np.flipud(np.fliplr(kernel))
    assert image.shape[-1]!=kernel.shape[-1]
    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[1]
    yKernShape = kernel.shape[2]
    zKernShape = kernel.shape[3]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]
    zImgShape = image.shape[2]
    
    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    zOutput = int(out_channels)
    output = np.zeros((xOutput, yOutput,zOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2,image.shape[2]))
        imagePadded[int(padding):int(-1 * padding),int(padding):int(-1 * padding),:] = image
    else:
        imagePadded = image
    # Iterate through image
   
    for z in range(out_channels):
        n=0 #y axis at output
        for y in range(image.shape[1]):
            m=0 #x axis at output
            # Exit Convolution
            if y > image.shape[1] - yKernShape:
                break
            # Only Convolve if y has gone down by the specified Strides
            if y % strides == 0:

                for x in range(image.shape[0]):
                    # Go to next row once kernel is out of bounds
                    if x > image.shape[0] - xKernShape:
                        break
                    try:
                        # Only Convolve if x has moved by the specified Strides
                        if x % strides == 0:
                            output[m, n,z] = (
                                kernel[z,:,:,:] * imagePadded[x: x + xKernShape, y: y + yKernShape,:]).sum()   
                            m+=1
                    except:
                        break
                n+=1    
                
    return output


if __name__ == '__main__':
    # Grayscale Image
    image = processImage('a.jpg')
    cv2.imshow('gray', image)
    cv2.waitKey(0)

    # Edge Detection Kernel ,kernel shape is [5,3,3,3]
    kernel = np.array([
                       [[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
                       [[-1, -1, -1], [-1, 0, -1], [0, 1, 1]],
                       [[1, 1, 1], [-1, 0, -1],[-1,-1,-1]]],

                       [[[1, 0, -1], [1, 0, -1], [1, 0, -1]],
                       [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
                       [[1, 0, -1], [1, 0, -1], [1, 0, -1]]],

                       [[[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                       [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                       [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]],

                       [[[-1, 0, -1], [-1, 8, -1], [-2, -1, -1]],
                       [[-1, -2, -1], [4, 0, -1], [0, -1, 1]],
                       [[1, 1, -1], [-1, 0, 1],[-1,-1,1]]],

                       [[[-1, -1, -1], [-1, 8, 1], [1, -3, -4]],
                       [[-1, -1, -1], [-1, 0, -1], [0, 10, 11]],
                       [[1, 1, 1], [-1, 0, -1],[1,1,-1]]]
                       ]
                       )
    # Convolve and Save Output
    output = convolve2D(image, kernel, padding=2,strides=2)
    cv2.imshow('feature_0', output[:,:,0])
    cv2.imshow('feature_1', output[:,:,1])
    cv2.imshow('feature_2', output[:,:,2])
    cv2.imshow('feature_3', output[:,:,3])
    cv2.imshow('feature_4', output[:,:,4])

    cv2.imshow('ori', image)
    cv2.waitKey(0)