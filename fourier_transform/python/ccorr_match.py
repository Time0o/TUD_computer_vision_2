#!/usr/bin/env python3

import argparse
import sys
from typing import List, Tuple

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

FINE_GRAINED_SALIENCY_THRESHOLD = 100

CCORR_MIN_PEAK = 0.5
CCORR_MAX_SLACK = 0.2
CCORR_MAX_LOCAL_SURROUNDING_RADIUS = 1


def autocrop(img: np.ndarray) -> np.ndarray:
    saliency = cv.saliency.StaticSaliencyFineGrained_create()
    success, saliency_map = saliency.computeSaliency(img)

    if not success:
        raise ValueError("failed to create saliency map")

    saliency_map /= (saliency_map.max() / 255)
    saliency_map = saliency_map.astype(np.uint8)

    _, thresh = cv.threshold(
        saliency_map, FINE_GRAINED_SALIENCY_THRESHOLD, 255, cv.THRESH_BINARY)

    _, cnt, _ = cv.findContours(
        thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not cnt:
        raise ValueError("failed to find object bounding rectangle")

    x, y, w, h = cv.boundingRect(cnt[0])

    return img[y:(y + h), x:(x + w)]


def template_match_ccorr(
        img: np.ndarray,
        template: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:

    ccorr = cv.matchTemplate(img, template, cv.TM_CCORR_NORMED)

    pad_top = template.shape[0] // 2
    pad_bottom = template.shape[0] - pad_top
    pad_left = template.shape[1] // 2
    pad_right = template.shape[1] - pad_left

    ccorr = cv.copyMakeBorder(ccorr,
                              pad_top, pad_bottom,
                              pad_left, pad_right,
                              cv.BORDER_CONSTANT, value=0)

    ccorr_min = ccorr.min()

    peaks = np.argwhere((ccorr > max(ccorr.max() - CCORR_MAX_SLACK, 0))
                        & (ccorr < ccorr.max() + CCORR_MAX_SLACK))

    peaks = [tuple(p) for p in peaks]

    local_maxima = []
    for peak in peaks:
        center = ccorr[peak[0], peak[1]]
        if center < ccorr_min + CCORR_MIN_PEAK:
            continue

        local_maximum = True
        r = CCORR_MAX_LOCAL_SURROUNDING_RADIUS
        for row in range(peak[0] - r, peak[0] + r + 1):
            for col in range(peak[1] - r, peak[1] + r + 1):
                if (row, col) == peak:
                    continue

                if ccorr[row, col] >= center:
                    local_maximum = False
                    break

            if not local_maximum:
                break

        if local_maximum:
            local_maxima.append((int(peak[0]), int(peak[1])))

    return ccorr, local_maxima


if __name__ == '__main__':
    # parse arguments
    def formatter_class(prog):
        return argparse.RawTextHelpFormatter(prog, max_help_position=80)

    parser = argparse.ArgumentParser(
        description="Locate image patches in an input image via cross-correlation.",
        formatter_class=formatter_class)

    parser.add_argument('image',
                        metavar='IMAGE',
                        help="input image in which to find matches")

    parser.add_argument('templates',
                        nargs='+',
                        metavar='TEMPLATE',
                        help="image patches to locate in input image")

    parser.add_argument('--auto-crop-templates',
                        action='store_true',
                        help="automatically crop templates to ROI")

    args = parser.parse_args()

    # read input image and templates
    def read_image(filename: str) -> np.ndarray:
        img = cv.imread(filename, flags=cv.IMREAD_GRAYSCALE)

        if img is None:
            print("Failed to read image file '{}'", file=sys.stderr)
            sys.exit(1)

        return img

    img = read_image(args.image)

    templates = [read_image(filename) for filename in args.templates]

    # auto-crop templates to ROI's
    if args.auto_crop_templates:
        templates = [autocrop(template) for template in templates]

    # perform template matching
    if len(templates) == 1:
        _, axes = plt.subplots(1, 2)

        ccorr, matches = template_match_ccorr(img, templates[0])

        axes[0].set_title("Template Match Locations")
        axes[0].imshow(img, cmap='gray')

        axes[1].set_title("Cross Correlation Result")
        axes[1].imshow(ccorr)

        if matches:
            print('{}: {}'.format(args.templates[0], matches))

            matches_rows, matches_cols = zip(*matches)
            axes[1].scatter(matches_cols, matches_rows, marker='x', color='r')
            axes[0].scatter(matches_cols, matches_rows, marker='x', color='r')

    else:
        plt.imshow(img, cmap='gray')

        for i, template in enumerate(templates):
            _, matches = template_match_ccorr(img, template)

            if matches:
                print('{}: {}'.format(args.templates[i], matches))

                matches_rows, matches_cols = zip(*matches)

                l = args.templates[i]
                plt.scatter(matches_cols, matches_rows, marker='x', label=l)

            plt.legend()

    plt.show()
