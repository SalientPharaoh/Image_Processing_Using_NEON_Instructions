import numpy as np
from WTHE import *
import cv2
import os

filename = "../images/Plane.jpg"

# load the image
image = cv2.imread(filename)
name, ext = os.path.splitext(filename)

# convert the image to HSV
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
image_v = image_hsv[:, :, 2].copy()

# apply the weighted thresholded histogram equalization
image_v_heq, _ = imWTHeq(image_v, r=0.5, v=0.5)

# apply the histogram equalization to the image
image_hsv[:, :, 2] = image_v_heq.copy()

# convert the image back to BGR
image_heq = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

cv2.imwrite(f"{name}-imWeightedThresholdedheq{ext}", image_heq)
