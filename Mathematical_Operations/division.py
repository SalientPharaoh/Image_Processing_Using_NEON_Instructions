import cv2
import numpy as np

def divide_image(input_image, divisor):
    # Divide pixel values by the divisor
    output_image = np.clip(input_image / divisor, 0, 255).astype(np.uint8)
    return output_image

def main():
    input_image = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        print('Error loading image!')
        return

    result_image = divide_image(input_image, 2.0)

    cv2.imshow('Divided Image', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
