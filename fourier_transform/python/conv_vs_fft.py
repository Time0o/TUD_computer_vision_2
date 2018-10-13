#!/usr/bin/env python3

import argparse
import os
import sys
from collections import OrderedDict
from typing import Callable, Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

IMAGE_DTYPES = OrderedDict([
    ('uint8', np.uint8),
    ('uint16', np.uint8),
    ('uint32', np.uint8),
    ('uint64', np.uint8),
])

DEFAULT_IMAGE_WIDTH = 1000
DEFAULT_IMAGE_HEIGHT = 1000
DEFAULT_IMAGE_DTYPE = next(iter(IMAGE_DTYPES))

DEFAULT_KERNEL_RADIUS_MIN = 5
DEFAULT_KERNEL_RADIUS_MAX = 100
DEFAULT_KERNEL_RADIUS_STEP = 5
DEFAULT_KERNEL_SIGMA = 10.0

DEFAULT_RUNS = 10


def random_array(size: Tuple[int, int], dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        dtype_max = np.iinfo(dtype).max
        return np.random.randint(dtype_max + 1, size=size, dtype=dtype)

    if np.issubdtype(dtype, np.floating):
        dtype_max = np.finfo(dtype).max
        return np.random.uniform(high=dtype_max, size=size).astype(dtype)

    raise ValueError("invalid datatype: {}".format(dtype))


def convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv.filter2D(img, -1, kernel)


def fft_pad(mat: np.ndarray, min: Tuple[int, int] = None):
    if min is not None:
        initial_width = max(min[0], mat.shape[0])
        initial_height = max(min[1], mat.shape[1])
    else:
        initial_width, initial_height = mat.shape

    dft_size = (cv.getOptimalDFTSize(initial_width),
                cv.getOptimalDFTSize(initial_height))

    row_pad = dft_size[0] - mat.shape[0]
    row_pad_top = row_pad // 2
    row_pad_bottom = row_pad - row_pad_top

    col_pad = dft_size[1] - mat.shape[1]
    col_pad_left = col_pad // 2
    col_pad_right = col_pad - col_pad_left

    res = cv.copyMakeBorder(mat,
                            row_pad_top, row_pad_bottom,
                            col_pad_left, col_pad_right,
                            cv.BORDER_CONSTANT, value=0)

    return res, [row_pad_top, row_pad_bottom, col_pad_left, col_pad_right]


def fft_apply(mat: np.ndarray):
    mat_complex = cv.merge([mat.astype(np.float32),
                            np.zeros_like(mat, dtype=np.float32)])

    cv.dft(mat_complex, dst=mat_complex)

    return mat_complex


def ifft_apply(mat: np.ndarray):
    return cv.dft(mat, flags=cv.DFT_INVERSE)[:, :, 0]


def fft_filter(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # apply FFT to image
    img_padded, img_pads = fft_pad(img)
    img_fft = fft_apply(img_padded)

    # apply FFT to kernel
    kernel_padded, _ = fft_pad(kernel, min=img.shape)
    kernel_fft = fft_apply(kernel_padded)

    # apply kernel to image
    kernel_re, kernel_im = cv.split(kernel_fft)
    kernel_magnitude = cv.magnitude(kernel_re, kernel_im)

    img_fft[:, :, 0] *= kernel_magnitude
    img_fft[:, :, 1] *= kernel_magnitude

    # transform back to spatial domain
    img_filtered = ifft_apply(img_fft)

    # remove padding and return result
    return img_filtered[img_pads[0]:-img_pads[1], img_pads[2]:-img_pads[3]]


def profile(func: Callable[[np.ndarray, np.ndarray], np.ndarray],
            img: np.ndarray,
            kernel: np.ndarray,
            runs: int) -> Tuple[float, float, float]:
    res = []

    for _ in range(runs):
        t0 = cv.getTickCount()

        func(img, kernel)

        res.append((cv.getTickCount() - t0) * 1000 / cv.getTickFrequency())

    return sum(res) / len(res), res[len(res) // 2], np.std(res)


if __name__ == '__main__':
    # parse arguments
    def formatter_class(prog):
        return argparse.RawTextHelpFormatter(prog, max_help_position=80)

    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION...]",
        description=str("Compare runtimes of gauss filtering by convolution\n"
                        "in the spatial domain and multiplication in the\n"
                        "frequency domain for different filter kernel sizes.\n"
                        "\n"
                        "Either specify a concrete input image on which\n"
                        "profiling should be performed using --image or\n"
                        "alternatively specify --random-image."),
        formatter_class=formatter_class)

    parser.add_argument('-s', '--silent',
                        action='store_true',
                        help="do not write progress to stdout")

    parser.add_argument('--image',
                        help="input image")

    parser.add_argument('--random-image',
                        action='store_true',
                        help="randomly generate input image")

    image_width_help = "random input image width (default {})"
    parser.add_argument('--image-width',
                        type=int,
                        help=image_width_help.format(DEFAULT_IMAGE_WIDTH))

    image_height_help = "random input image height (default {})"
    parser.add_argument('--image-height',
                        type=int,
                        help=image_height_help.format(DEFAULT_IMAGE_HEIGHT))

    image_dtype_help = "random input image datatype (default {})"
    parser.add_argument('--image-dtype',
                        choices=IMAGE_DTYPES.keys(),
                        help=image_dtype_help.format(DEFAULT_IMAGE_DTYPE))

    kernel_radius_min_help = "minimal filter kernel radius (default {})"
    parser.add_argument('--kernel-radius-min',
                        default=DEFAULT_KERNEL_RADIUS_MIN,
                        type=int,
                        help=kernel_radius_min_help.format(
                            DEFAULT_KERNEL_RADIUS_MIN))

    kernel_radius_max_help = "maximal filter kernel radius (default {})"
    parser.add_argument('--kernel-radius-max',
                        default=DEFAULT_KERNEL_RADIUS_MAX,
                        type=int,
                        help=kernel_radius_max_help.format(
                            DEFAULT_KERNEL_RADIUS_MAX))

    kernel_radius_step_help = "filter kernel radius step size (default {})"
    parser.add_argument('--kernel-radius-step',
                        default=DEFAULT_KERNEL_RADIUS_STEP,
                        type=int,
                        help=kernel_radius_step_help.format(
                            DEFAULT_KERNEL_RADIUS_STEP))

    runs_help = "profiling runs per kernel (default {})"
    parser.add_argument('--runs',
                        default=DEFAULT_RUNS,
                        type=int,
                        help=runs_help.format(DEFAULT_RUNS))

    args = parser.parse_args()

    if args.image is not None:
        if args.random_image:
            err = "can not simultaneously specify --image and --random-image"
            raise ValueError(err)

        if args.image_width is not None:
            print("Warning: --image specified, ignoring --image-width",
                  file=sys.stderr)

        if args.image_height is not None:
            print("Warning: --image specified, ignoring --image-height",
                  file=sys.stderr)

        if args.image_dtype is not None:
            print("Warning: --image specified, ignoring --image-dtype",
                  file=sys.stderr)
    else:
        if not args.random_image:
            parser.print_usage(sys.stderr)

            fmt = "{}: error: either --image or --random-image must be specified"
            print(fmt.format(os.path.basename(__file__)), file=sys.stderr)

            sys.exit(1)

        if args.image_width is None:
            args.image_width = DEFAULT_IMAGE_WIDTH

        if args.image_height is None:
            args.image_height = DEFAULT_IMAGE_HEIGHT

        if args.image_dtype is None:
            args.image_dtype = DEFAULT_IMAGE_DTYPE

    # obtain test-image
    if args.image is not None:
        img = cv.imread(args.image, flags=cv.IMREAD_GRAYSCALE)

        if img is None:
            print("Failed to read image file '{}'", file=sys.stderr)
            sys.exit(1)
    else:
        dtype = IMAGE_DTYPES[args.image_dtype]
        img = random_array((args.image_width, args.image_height), dtype)

    # profile convolution and FFT
    rmin = args.kernel_radius_min
    rmax = args.kernel_radius_max
    rstep = args.kernel_radius_step

    radii = list(range(rmin, rmax + 1, rstep))

    results_convolution = []
    results_fft = []

    for i, r in enumerate(radii):
        w = 2 * r + 1

        if not args.silent:
            fmt = "({}/{}) profiling {w}x{w} kernel..."
            print(fmt.format(i + 1, len(radii), w=w))

        gaussian1D = cv.getGaussianKernel(w, DEFAULT_KERNEL_SIGMA)
        kernel = gaussian1D * gaussian1D.T

        results_convolution.append(
            profile(convolve, img, kernel, args.runs))

        results_fft.append(
            profile(fft_filter, img, kernel, args.runs))

        if i == len(radii) - 1:
            img_convolved = convolve(img, kernel)
            img_fft_filtered = fft_filter(img, kernel)

    # display results
    if args.image:
        fig, axes = plt.subplots(2, 2)
    else:
        fig, axes = plt.subplots(1, 2)

    fig.set_size_inches(12, 7)

    # convolution
    title_conv = "Convolution ({}x{})".format(img.shape[0], img.shape[1])
    t_conv_avg, t_conv_med, t_conv_std = zip(*results_convolution)

    if args.image:
        axes[0, 0].imshow(img_convolved, cmap='gray')

        post_title = ", {w}x{w} kernel".format(w=2 * radii[-1] + 1)
        axes[0, 0].set_title(title_conv + post_title)

    ax_conv_data = axes[0, 1] if args.image else axes[0]
    ax_conv_data.set_title(title_conv)

    ax_conv_data.set_xlabel("Kernel Radius (px)")
    ax_conv_data.set_ylabel("Execution Time (ms)")

    ax_conv_data.plot(radii, t_conv_avg,
                      label="average ({} runs)".format(args.runs))

    ax_conv_data.errorbar(radii, t_conv_med, t_conv_std,
                          capsize=5,
                          elinewidth=1,
                          label="median and stddev ({} runs)".format(args.runs))

    ax_conv_data.legend()

    # FFT
    title_fft = "FFT ({}x{})".format(img.shape[0], img.shape[1])
    t_fft_avg, t_fft_med, t_fft_std = zip(*results_fft)

    if args.image:
        axes[1, 0].imshow(img_fft_filtered, cmap='gray')

        post_title = ", {w}x{w} kernel".format(w=2 * radii[-1] + 1)
        axes[1, 0].set_title(title_fft + post_title)

    ax_fft_data = axes[1, 1] if args.image else axes[1]
    ax_fft_data.set_title(title_fft)

    ax_fft_data.set_xlabel("Kernel Radius (px)")
    ax_fft_data.set_ylabel("Execution Time (ms)")

    ax_fft_data.plot(radii, t_fft_avg,
                     label="average ({} runs)".format(args.runs))

    ax_fft_data.errorbar(radii, t_fft_med, t_fft_std,
                         capsize=5,
                         elinewidth=1,
                         label="median and stddev ({} runs)".format(args.runs))

    ax_fft_data.legend()

    # show plots
    plt.tight_layout()

    plt.show()
