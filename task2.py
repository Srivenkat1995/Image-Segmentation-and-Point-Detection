import cv2 

import numpy as np 

from scipy import signal 

import math

import matplotlib.pyplot as plt

if __name__ == "__main__":

    gauss_blur_filter = [[0 for x in range(3)] for y in range(3)]

    gauss_blur_filter[0][0] = 1/16 
    gauss_blur_filter[0][1] = 1/8
    gauss_blur_filter[0][2] = 1/16
    gauss_blur_filter[1][0] = 1/8
    gauss_blur_filter[1][1] = 1/4
    gauss_blur_filter[1][2] = 1/8
    gauss_blur_filter[2][0] = 1/16
    gauss_blur_filter[2][1] = 1/8
    gauss_blur_filter[2][2] = 1/16

    image = cv2.imread('point.jpg',0)

    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype = np.float)

    replicate = cv2.copyMakeBorder(image,20,20,20,20,cv2.BORDER_REPLICATE)

    resultant_image = cv2.blur(replicate,(5,5))

    cv2.imwrite('gauss-blue.jpg',resultant_image)

    resultant_image_1 = signal.convolve2d(image,kernel,'same')

    rows,columns = resultant_image_1.shape



    for i in range(rows):
        for j in range(columns):
            
            resultant_image_1[i][j] = abs(resultant_image_1[i][j])

    cv2.imwrite('mask-application.jpg',resultant_image_1)

    print(resultant_image_1.max())

    for i in range(rows):
        for j in range(columns):

            if resultant_image_1[i][j] >= 2024:

                print(i,j)

            else:

                resultant_image_1[i][j] = 0   



    cv2.imwrite('point-detection.jpg',resultant_image_1) 

    image_segment = cv2.imread('segment.jpg',0)

    rows,columns = image_segment.shape

    '''x = np.zeros(255)

    y = np.arange(0,255,1)
    for i in range(rows):
        for j in range(columns):
            if image_segment[i][j] != 0:
                x[image_segment[i][j]] += 1

    hist, bins = np.histogram(x, bins=y)
    width = 1.0 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)

    #plt.plot(y,x)
    #plt.bar(np.arange(len(y)),y)
    plt.show()'''

    for i in range(rows):
        for j in range(columns):

            if image_segment[i][j] > 208 or image_segment[i][j] < 200 :
                image_segment[i][j] = 0
    
    
    cv2.imwrite('segemented.jpg',image_segment)            
             