import cv2
import numpy as np

class NLMDenoising:
    def __init__(self, template_window=7, search_window=21, h_param=10.0, multi_thread=True):
        self.template_window_size = template_window
        self.search_window_size = search_window
        self.h = h_param
        self.use_multi_threading = multi_thread

    def denoise(self, input_image):
        assert input_image is not None and input_image.shape[2] <= 4
        if len(input_image.shape) == 2:
            # Grayscale image processing
            return cv2.fastNlMeansDenoising(input_image, None, self.h, self.template_window_size, self.search_window_size)
        else:
            # Color image processing
            return cv2.fastNlMeansDenoisingColored(input_image, None, self.h, self.h, self.template_window_size, self.search_window_size)

    def denoise_custom(self, input_image):
        assert input_image is not None and input_image.shape[2] <= 4
        input_float = input_image.astype(np.float32) / 255.0
        output = input_float.copy()

        half_template = self.template_window_size // 2
        half_search = self.search_window_size // 2
        h_squared = self.h * self.h

        for i in range(half_template, input_float.shape[0] - half_template):
            for j in range(half_template, input_float.shape[1] - half_template):
                patch = input_float[i - half_template:i + half_template + 1, j - half_template:j + half_template + 1]
                weight_sum = 0
                pixel_sum = np.zeros(3)

                for si in range(-half_search, half_search + 1):
                    for sj in range(-half_search, half_search + 1):
                        ni, nj = i + si, j + sj
                        if ni < half_template or ni >= input_float.shape[0] - half_template or nj < half_template or nj >= input_float.shape[1] - half_template:
                            continue

                        neighbor_patch = input_float[ni - half_template:ni + half_template + 1, nj - half_template:nj + half_template + 1]
                        distance = np.sum((patch - neighbor_patch) ** 2) / (self.template_window_size ** 2)
                        weight = np.exp(-distance / h_squared)

                        weight_sum += weight
                        pixel_sum += weight * input_float[ni, nj]

                output[i, j] = np.clip(pixel_sum * 255.0 / weight_sum, 0, 255)

        return output.astype(np.uint8)

    def compare_implementations(self, input_image):
        opencv_result = self.denoise(input_image)
        custom_result = self.denoise_custom(input_image)

        mean_diff = np.mean(np.abs(opencv_result - custom_result))
        print(f"Average difference between implementations: {mean_diff}")

        cv2.imshow("Original", input_image)
        cv2.imshow("OpenCV NLM", opencv_result)
        cv2.imshow("Custom NLM", custom_result)
        cv2.waitKey(0)

# Example usage
if __name__ == '__main__':
    # Load an image
    image = cv2.imread('/home/jatayu/MyFiles/Samsung_Prism_NEON/images/input_images/BM3D/Basic3.jpg')  # Replace with actual image path
    nlm = NLMDenoising()
    nlm.compare_implementations(image)
