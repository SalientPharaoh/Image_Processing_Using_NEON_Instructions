import cv2
import numpy as np

def multiply_image(input_image, factor):
    # Multiply pixel values by the factor
    output_image = np.clip(input_image * factor, 0, 255).astype(np.uint8)
    return output_image

def main():
    input_image = cv2.imread('car.jpg', cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        print('Error loading image!')
        return

    result_image = multiply_image(input_image, 1.5)

    cv2.imshow('Multiplied Image', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
