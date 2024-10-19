import numpy as np
from numba import njit, prange

import kernels
import cv2 as cv


def display_image(title, image):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


@njit(parallel=True)
def convolve_numba(padded_img: np.ndarray, conv_img: np.ndarray, kernel: np.ndarray, k_size: int,
                   channels: int):
    for c in prange(channels):
        for i in range(k_size, conv_img.shape[0] - k_size):
            for j in range(k_size, conv_img.shape[1] - k_size):
                region = padded_img[i - k_size: i + k_size + 1, j - k_size: j + k_size + 1, c]
                conv_img[i, j, c] = np.sum(region * kernel)


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    conv_img = np.zeros_like(image, dtype=np.float32)
    p_size = kernel.shape[0] // 2

    if image.ndim == 2:
        pad_width = ((p_size, p_size), (p_size, p_size))
    elif image.ndim == 3:
        pad_width = ((p_size, p_size), (p_size, p_size), (0, 0))
    else:
        raise ValueError("Unsupported image dimensions.")

    padded_img = np.pad(image, pad_width, mode='constant')
    k_size = kernel.shape[0] // 2
    channels = conv_img.shape[2]
    convolve_numba(padded_img, conv_img, kernel, k_size, channels)

    conv_img = np.clip(conv_img, 0, 255).astype(np.uint8)
    display_image('Convolved Image', conv_img)

    return np.clip(conv_img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    gauss_filter = kernels.gaussian_kernel(3, 3)
    box_filter = kernels.box_kernel(101)
    img = cv.imread('data/Gura_Portitei_Scara_020.jpg', cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Image not found. Check the file path.")
    convolve(img, gauss_filter)
