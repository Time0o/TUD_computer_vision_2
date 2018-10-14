#!/usr/bin/env python3

import argparse
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def fft_apply(img: np.ndarray) -> np.ndarray:
    dft_size = (cv.getOptimalDFTSize(img.shape[0]),
                cv.getOptimalDFTSize(img.shape[1]))

    img_padded = cv.copyMakeBorder(img,
                                   0, dft_size[0] - img.shape[0],
                                   0, dft_size[1] - img.shape[1],
                                   cv.BORDER_CONSTANT, value=0)

    img_padded = img_padded.astype(np.float32)

    img_complex = cv.merge([img_padded, np.zeros_like(img_padded)])

    return cv.dft(img_complex, dst=img_complex)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('image',
                        metavar='IMAGE',
                        help="input image")

    args = parser.parse_args()

    # read input image
    img = cv.imread(args.image, flags=cv.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to read image file '{}'".format(args.image),
              file=sys.stderr)

        sys.exit(1)

    # apply FFT
    img_fft = fft_apply(img)

    # obtain FFT magnitude image
    img_re, img_im = cv.split(img_fft)
    img_magnitude = cv.magnitude(img_re, img_im)

    img_magnitude += 1.0
    img_magnitude = cv.log(img_magnitude)

    img_magnitude /= (img_magnitude.max() / 255)
    img_magnitude = img_magnitude.astype(np.uint8)

    r2, c2 = img_magnitude.shape[0] // 2, img_magnitude.shape[1] // 2

    tl = img_magnitude[:r2, :c2].copy()
    tr = img_magnitude[r2:, :c2].copy()
    bl = img_magnitude[:r2, c2:].copy()
    br = img_magnitude[r2:, c2:].copy()

    img_magnitude[:r2, :c2], img_magnitude[r2:, c2:] = br, tl
    img_magnitude[r2:, :c2], img_magnitude[:r2, c2:] = bl, tr

    # visualize result
    _, axes = plt.subplots(1, 2)

    axes[0].set_title("Spatial Domain")
    axes[0].imshow(img, cmap='gray')

    axes[1].set_title("Frequency Domain (Logarithmic Scale)")
    axes[1].imshow(img_magnitude, cmap='gray')

    plt.show()