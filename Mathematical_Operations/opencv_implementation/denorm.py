import cv2
import numpy as np

def denormalize_image(input_image):
    # Denormalize pixel values assuming they are in range [0, 1]
    output_image = np.clip(input_image * 255.0, 0, 255).astype(np.uint8)
    return output_image

def main():
    input_image = cv2.imread('example_normalized.jpg', cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        print('Error loading image!')
        return

    result_image = denormalize_image(input_image)

    cv2.imshow('Denormalized Image', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
