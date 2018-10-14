#!/usr/bin/env python3

import argparse
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

MIN_PEAK_NOISE_DISTANCE = 1.0
MAX_PEAK_SYMMETRY_SLACK_MANHATTEN = 1


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

    # obtain frequency magnitude image
    img_re, img_im = cv.split(img_fft)
    img_magnitude = cv.magnitude(img_re, img_im)

    r2, c2 = img_magnitude.shape[0] // 2, img_magnitude.shape[1] // 2

    tl = img_magnitude[:r2, :c2].copy()
    tr = img_magnitude[r2:, :c2].copy()
    bl = img_magnitude[:r2, c2:].copy()
    br = img_magnitude[r2:, c2:].copy()

    img_magnitude[:r2, :c2], img_magnitude[r2:, c2:] = br, tl
    img_magnitude[r2:, :c2], img_magnitude[:r2, c2:] = bl, tr

    # determine frequency domain quadrant peaks
    img_magnitude[r2, c2] = 0.0

    _, _, _, max_loc_tl = cv.minMaxLoc(img_magnitude[:r2, :c2])
    _, _, _, max_loc_tr = cv.minMaxLoc(img_magnitude[:r2, c2:])
    _, _, _, max_loc_bl = cv.minMaxLoc(img_magnitude[r2:, :c2])
    _, _, _, max_loc_br = cv.minMaxLoc(img_magnitude[r2:, c2:])

    max_loc_tr = (max_loc_tr[0] + tl.shape[1], max_loc_tr[1])

    max_loc_bl = (max_loc_bl[0], max_loc_bl[1] + tl.shape[0])

    max_loc_br = (max_loc_br[0] + tl.shape[1],
                  max_loc_br[1] + tl.shape[0])

    # check whether detected peaks are plausible
    max_avg_diag1 = (img_magnitude[max_loc_tl[1], max_loc_tl[0]] +
                     img_magnitude[max_loc_br[1], max_loc_br[0]]) / 2.0

    max_avg_diag2 = (img_magnitude[max_loc_tr[1], max_loc_tr[0]] +
                     img_magnitude[max_loc_bl[1], max_loc_bl[0]]) / 2.0

    max_avg_dist = abs(max_avg_diag1 - max_avg_diag2)

    if max_avg_diag1 > max_avg_diag2:
        peak1, peak2 = max_loc_tl, max_loc_br
    else:
        peak1, peak2 = max_loc_tr, max_loc_bl

    peak1_mirrored = (img_magnitude.shape[1] - peak1[0],
                      img_magnitude.shape[0] - peak1[1])

    peak_dist = abs(peak1_mirrored[0] - peak2[0]) + \
                abs(peak1_mirrored[1] - peak2[1])

    if max_avg_dist < MIN_PEAK_NOISE_DISTANCE or \
       peak_dist > MAX_PEAK_SYMMETRY_SLACK_MANHATTEN:

        print("Failed to determine text orientation", file=sys.stderr)
        sys.exit(1)

    # determine line orientation and spacing
    img_marked = img.copy()
    img_marked = cv.cvtColor(img_marked, cv.COLOR_GRAY2BGR)

    line_angle = np.arctan2(peak2[0] - peak1[0], peak2[1] - peak1[0])
    line_angle *= 180 / np.pi

    line_dist = np.sqrt((peak2[0] - peak1[0])**2 + (peak2[1] - peak1[1])**2)

    line_row = 0.0 # TODO
    while line_row < img_marked.shape[0]:
        line_box_width = img_marked.shape[1]
        line_box_height = line_dist / 2

        line_box_x = 0 + line_box_width / 2
        line_box_y = line_row + line_box_height / 2

        line_box = cv.boxPoints(((line_box_x, line_box_y),
                                 (line_box_width, line_box_height),
                                 line_angle))

        line_box = line_box.astype(np.int)

        cv.drawContours(img_marked, [line_box], 0, (255, 0, 0))

        line_row += line_dist / 2

    # convert frequency magnitude image to logarithmic grayscale
    img_magnitude = cv.log(img_magnitude)

    img_magnitude /= (img_magnitude.max() / 255)
    img_magnitude = img_magnitude.astype(np.uint8)

    # visualize result
    _, axes = plt.subplots(1, 2)

    axes[0].set_title("Spatial Domain")
    axes[0].imshow(img_marked, cmap='gray')

    axes[1].set_title("Frequency Domain (Logarithmic Scale)")
    axes[1].imshow(img_magnitude, cmap='gray')

    axes[1].scatter(peak1[0], peak1[1], marker='+', color='r')
    axes[1].scatter(peak2[0], peak2[1], marker='+', color='r')

    plt.show()
