#!/usr/bin/env python3

import argparse
import sys
from typing import Optional

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

DETECTION_THRESHOLD = 100
BLUR_KERNEL_SIZE = 500
BLUR_SIGMA = 20.0


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('image',
                        metavar='IMAGE')

    args = parser.parse_args()

    # read input image
    img = cv.imread(args.image, flags=cv.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to read image file '{}'".format(args.image),
              file=sys.stderr)

        sys.exit(1)

    # locate sinusoidal in input image
    line_rows = np.empty((img.shape[1]), dtype=np.int)

    holes = []
    hole_first = None

    for col in range(img.shape[1]):
        argmax = img[:, col].argmax()

        if img[argmax, col] > DETECTION_THRESHOLD:
            if hole_first is not None:
                holes.append((hole_first, col - 1))
                hole_first = None

            line_rows[col] = argmax

        elif hole_first is None:
            hole_first = col

    # interpolate holes in sinusoidal
    #TODO: what if first/last segments are holes?
    for hole_first, hole_last in holes:
        interp_last = line_rows[hole_first - 1]
        interp_next = line_rows[hole_last + 1]

        def interp(col):
            m = (interp_next - interp_last) / (hole_last - hole_first + 2)
            b = interp_last - m * (hole_first - 1)

            return m * col + b

        for col in range(hole_first, hole_last + 1):
            line_rows[col] = interp(col)

    # smooth column positions
    def interpolate_position(w0: int, w1: int, w2: int) -> Optional[float]:
        w = w0 + w2 - 2 * w1

        if w == 0:
            return None

        return (w0 - w2) / (2 * w)

    line_cols_smoothed = np.arange(img.shape[1], dtype=np.float)

    for col in range(1, img.shape[1] - 1):
        w0, w1, w2 = line_rows[(col - 1):(col + 2)]

        pos = interpolate_position(w0, w1, w2)
        if pos is not None:
            line_cols_smoothed[col] += pos

    # eliminate duplicate column positions
    line_cols_unique = np.unique(line_cols_smoothed)
    line_rows_smoothed = np.empty_like(line_cols_unique)

    run_value, run_length, run_acc = -1.0, 0, 0.0

    i, j = 0, 0
    for i in range(len(line_cols_smoothed)):
        col = line_cols_smoothed[i]

        if col == run_value:
            run_length += 1
            run_acc += line_rows[i]
        else:
            if i == 0:
                line_rows_smoothed[j] = line_rows[i]
            else:
                line_rows_smoothed[j] = run_acc / run_length

            j += 1

            run_value = col
            run_length = 1
            run_acc = line_rows[i]

    # oversample to create equidistant spacing
    min_dist = 1.0
    for i in range(len(line_cols_unique) - 1):
        dist = line_cols_unique[i + 1] - line_cols_unique[i]
        min_dist = min(min_dist, dist)

    oversampling_factor = int(1 / min_dist)
    oversampling_cols = int(line_cols_unique.max() * oversampling_factor) + 1

    line_cols_oversampled = np.linspace(0, img.shape[1] - 1, oversampling_cols)
    line_rows_oversampled = np.full(oversampling_cols, np.nan, dtype=np.float)

    for i in range(len(line_cols_unique)):
        j = int(line_cols_unique[i] * oversampling_factor)
        line_rows_oversampled[j] = line_rows_smoothed[i]

    row_left = line_rows_oversampled[0]
    for i in range(len(line_rows_oversampled)):
        row = line_rows_oversampled[i]

        if np.isnan(row):
            line_rows_oversampled[i] = row_left
        else:
            row_left = row

    # apply gaussian filter
    kernel = cv.getGaussianKernel(BLUR_KERNEL_SIZE, BLUR_SIGMA)
    kernel = kernel.reshape((BLUR_KERNEL_SIZE))

    r = BLUR_KERNEL_SIZE // 2
    line_rows_oversampled = np.pad(line_rows_oversampled, r, 'edge')
    line_rows_oversampled = np.convolve(line_rows_oversampled, kernel, mode='same')
    line_rows_oversampled = line_rows_oversampled[r:-r]

    # apply FFT
    line_fft = np.fft.fft(np.array(line_rows_oversampled))
    line_fft_magnitude = np.absolute(line_fft)

    # visualize result
    _, axes = plt.subplots(1, 2)

    axes[0].imshow(img, cmap='gray')

    axes[0].plot(line_cols_oversampled, line_rows_oversampled,
                 linestyle='None', marker='.', markersize=1, color='b')

    axes[1].semilogy(line_fft_magnitude)

    plt.show()
