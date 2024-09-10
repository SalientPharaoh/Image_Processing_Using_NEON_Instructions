import argparse
from argparse import RawTextHelpFormatter
import glob
from os import makedirs
from os.path import join, exists, basename, splitext

import cv2
from tqdm import tqdm

from exposure_enhancement import enhance_image_exposure

# load images
file = "./2.bmp"
images = [cv2.imread(file)]
# create save directory
directory = "."
# enhance images
for i, image in tqdm(enumerate(images), desc="Enhancing images"):
    gamma = 0.6
    lambda_ = 0.15
    sigma = 3
    bc = 1
    bs = 1
    be = 1
    eps = 1e-3
    useDUAL = True
    enhanced_image = enhance_image_exposure(image, gamma, lambda_, useDUAL,sigma=sigma, bc=bc, bs=bs, be=be, eps=eps)
    filename = basename(file)
    name, ext = splitext(filename)
    method = "DUAL" # change it to DUAL for using DUAL method or LIME for using LIME method
    corrected_name = f"{name}_{method}_g{gamma}_l{lambda_}{ext}"
    cv2.imwrite(join(directory, corrected_name), enhanced_image)