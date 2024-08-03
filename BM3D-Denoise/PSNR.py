import cv2
import math
import numpy

# calculates the peak signa, to noise ratio between two images
def PSNR(img1, img2):
    D = numpy.array(img1 - img2, dtype=numpy.int64) # pixel difference
    D[:, :] = D[:, :]**2 #square
    RMSE = D.sum()/img1.size # mean square error
    psnr = 10*math.log10(float(255.**2)/RMSE) # peak signal to noise ratio
    return psnr

if __name__ == "__main__":
    img1 = cv2.imread("original 2D4F.bmp", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("Basic2.jpg", cv2.IMREAD_GRAYSCALE)
    psnr = PSNR(img1, img2)
    print ("The PSNR between the two img of the two is %f" % psnr)

    img1 = cv2.imread("original 2D4F.bmp", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("Final2.jpg", cv2.IMREAD_GRAYSCALE)
    psnr = PSNR(img1, img2)
    print ("The PSNR between the two img of the two is %f" % psnr)

