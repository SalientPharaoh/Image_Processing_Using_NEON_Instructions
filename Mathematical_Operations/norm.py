import cv2
import numpy as np

def normalize_image(input_image):
    # Normalize pixel values to range [0, 255]
    output_image = np.clip(input_image / 255.0 * 255, 0, 255).astype(np.uint8)
    return output_image

def main():
    input_image = cv2.imread('img1.jpg', cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        print('Error loading image!')
        return

    result_image = normalize_image(input_image)

    cv2.imshow('Normalized Image', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
