# NEON implementation of mathematical operations for image processing

Welcome to the Image Processing Algorithms repository! This project contains implementations of various image processing algorithms and basic mathematical operations using both OpenCV and NEON. This repository is designed to facilitate research, development, and testing of image processing techniques.

## Table of Contents

- [NEON implementation of mathematical operations for image processing](#neon-implementation-of-mathematical-operations-for-image-processing)
  - [Table of Contents](#table-of-contents)
  - [Folder Structure](#folder-structure)
  - [Image Processing Algorithms](#image-processing-algorithms)
    - [BM3D-Denoise](#bm3d-denoise)
    - [LIME](#lime)
      - [DUAL](#dual)
    - [WTHE](#wthe)
  - [Mathematical Operations](#mathematical-operations)
  - [Images](#images)
  - [Review Slides](#review-slides)
  - [Getting Started](#getting-started)

## Folder Structure

Here's an overview of the folder structure in this repository:

```bash
Image processing algorithms/
|-- BM3D-Denoise/
|   |-- opencv_implementation/
|   |-- NEON_implementation/
|-- LIME/
|   |-- opencv_implementation/
|   |-- NEON_implementation/
|-- WTHE/
|   |-- opencv_implementation/
|   |-- NEON_implementation/
images/
|-- input_images/
|-- output_images/
mathematical operations/
|-- opencv_implementation/
|-- NEON_implementation/
review slides/
```

## Image Processing Algorithms

This repository contains implementations of the following image processing algorithms:

### BM3D-Denoise

BM3D (Block-Matching and 3D Filtering) is a widely-used algorithm for image denoising. It works by grouping similar blocks of image data, transforming them into a common space, and then applying a 3D filter.

[Reference Paper](https://jcst.ict.ac.cn/en/article/pdf/preview/10.1007/s11390-018-1859-7.pdf)

- **OpenCV Implementation**: Contains the OpenCV-based implementation of BM3D.
- **NEON Implementation**: Contains the ARM NEON-based implementation for optimized performance on ARM architectures.

### LIME

LIME (Low Light Image Enhancement) is a technique designed to improve the quality of images captured in low-light conditions. It aims to enhance brightness and contrast, making details more visible while reducing noise and preserving image quality.

[Reference Paper](https://ieeexplore.ieee.org/abstract/document/7782813)

- **OpenCV Implementation**: Contains the OpenCV-based implementation of LIME.
- **NEON Implementation**: Contains the ARM NEON-based implementation for optimized performance on ARM architectures.

#### DUAL

DUAL is an extension of LIME algorithm. DUAL also focuses on correction of the over-exposure in the image by inverting the illumination map and applying the LIME algorithm to the inverted map and reinverting the image to obtain the original image with corrected ovver-exposure.

[Reference Paper](https://ieeexplore.ieee.org/abstract/document/9494618)

### WTHE

WTHE (Weighted Threshold Histogram Equalization) is an advanced image enhancement technique that improves image contrast and brightness by adjusting the histogram based on weighted thresholds. This method enhances local contrast while preserving important details in the image.

[Reference Paper](https://www.researchgate.net/profile/Uma-Ramadass/publication/228887807_A_New_Approach_To_Image_Contrast_Enhancement_using_Weighted_Threshold_Histogram_Equalization_with_Improved_Switching_Median_Filter/links/53ee3d7a0cf2981ada175e63/A-New-Approach-To-Image-Contrast-Enhancement-using-Weighted-Threshold-Histogram-Equalization-with-Improved-Switching-Median-Filter.pdf)

- **OpenCV Implementation**: Contains the OpenCV-based implementation of WTHE.
- **NEON Implementation**: Contains the ARM NEON-based implementation for optimized performance on ARM architectures.

## Mathematical Operations

This section includes basic mathematical operations implemented using OpenCV and NEON. These operations are fundamental to many image processing tasks.

- **OpenCV Implementation**: Includes code for operations such as multiplication, division, normalization, and denormalization using OpenCV.
- **NEON Implementation**: Includes optimized implementations for the same operations using ARM NEON.

## Images

This folder contains images used for testing and demonstration purposes.

- **Input Images**: Contains sample images that are used as inputs for various algorithms.
- **Output Images**: Contains images produced as outputs from the algorithms, allowing for easy comparison and validation of results.

## Review Slides

This folder includes review meeting slides and reports. These documents provide insights into the progress, findings, and evaluations of the worklet.

## Getting Started

To get started with this repository, follow these steps:

1. **Clone the Repository**:
  
   ```bash
   git clone <repository_link>
   cd <repository_root>
   ```

2. **Install Dependencies**:
   Ensure you have the required dependencies installed. For OpenCV-based implementations, you need to install OpenCV. For NEON-based implementations, ensure your ARM environment is properly set up.

   ```bash
   sudo apt-get install libopencv-dev  #for linux based systems
   pip install python-opencv
   ```

3. **Build and Run**:
   Follow the instructions in each implementation folder to build and run the code. Specific build instructions may vary based on the implementation and environment.
