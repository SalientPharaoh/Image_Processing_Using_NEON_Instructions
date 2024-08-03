import numpy as np
import numba

def imhist(image):
    # flatten the image into 1D array
    # Convert the image into 256 bins histogram array (values between 0-255)
    hist, _ = np.histogram(image.reshape(1, -1), bins=256, range=(0, 255))
    return hist

def imWTHeq(image, Wout_list=np.zeros((10)), r=0.5, v=0.5):
    [h, w] = image.shape # get image dimensions
    PMF = imhist(image) / (h * w) # get the probability mass function of the image
    Pl = 1e-4 # lower threshold
    Pu = v * np.max(PMF) # upper threshold (scaled)
    Pwt = np.zeros_like(PMF) # weighted probability mass function
    for indx, pmf in enumerate(PMF):
        # if pmf is less than lower threshold, set the weighted probability to 0
        # if pmf is greater than upper threshold, set the weighted probability to upper threshold
        if pmf < Pl:
            Pwt[indx] = 0
        elif pmf > Pu:
            Pwt[indx] = Pu
        else:
            Pwt[indx] = (((pmf - Pl) / (Pu - Pl)) ** r) * Pu

    Cwt = np.cumsum(Pwt) # cumulative sum of weighted probability mass function
    Cwtn = Cwt / Cwt[-1] # normalize the sum to 0 to 1 range
    Win = len(np.where(PMF > 0)[0]) # get number of non zero pmfs
    Gmax = 1.5  # maximum gain factor (ideally between 1.5 to 2)

    Wout = min(255, Gmax * Win) # get the output weight based on gain and input weights

    if np.where(Wout_list > 0)[0].size == Wout_list.size:
        Wout = (np.sum(Wout_list) + Wout) / (1 + Wout_list.size) #if wout list already there, average with current wout values

    # histogram equalization
    F = image.copy().reshape(-1) # flatten image
    Ftilde = Wout * Cwtn[F] # apply normalized cumulative distribution function to the image

    # Mean adjustment
    Madj = np.mean(image) - np.mean(Ftilde) # image mean - cdf applied image mean
    Ftilde = Ftilde + Madj #adjustment

    # clipping and reshaping
    Ftilde = np.where(Ftilde >= 0, Ftilde, np.zeros_like(Ftilde)) # clip values less than 0 to 0
    Ftilde = np.where(Ftilde <= 255, Ftilde, 255 * np.ones_like(Ftilde)) # clip values greater than 255 to 255
    Ftilde = Ftilde.astype(np.uint8) # convert to uint8

    image_heq = Ftilde.reshape(h, w) # reshape to original image dimensions

    # returns the histogram equalized image and the output weight
    return image_heq, Wout
