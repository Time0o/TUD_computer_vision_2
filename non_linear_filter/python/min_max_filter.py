#!/usr/bin/env python3

import argparse

import cv2 as cv
import numpy as np

def extremum_filter(img: np.ndarray,
                    filtertype: str,
                    radius: int = 1,
                    method: str = 'naive'):

    assert len(img.shape) == 2 and img.dtype == np.uint8

    # filter width
    width = 2 * radius + 1

    # allocate filtered image
    img_filtered = np.empty_like(img)

    # pad original image
    if filtertype == 'min':
        func = cv.erode if method == 'opencv' else np.min

        img = np.pad(img, radius, mode='constant',
                     constant_values=np.iinfo(np.uint8).max)

    elif filtertype == 'max':
        func = cv.dilate if method == 'opencv' else np.max

        img = np.pad(img, radius, mode='constant',
                     constant_values=np.uint8(0))
    else:
      raise ValueError("unsupported filter type '{}'".format(filtertype))

    # apply filter
    if method == 'naive':
        # horizontal pass
        for col in range(img_filtered.shape[1]):
            for row in range(img_filtered.shape[0]):
                window = img[row:(row + width), col + radius]
                img_filtered[row, col] = func(window)

        img[radius:-radius, radius:-radius] = img_filtered

        # vertical pass
        for row in range(img_filtered.shape[0]):
            for col in range(img_filtered.shape[1]):
                window = img[row + radius, col:(col + width)]
                img_filtered[row, col] = func(window)

    elif method == 'queue':
        pass

    elif method == 'opencv':
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (width, width))
        img_filtered = func(img, kernel)[radius:-radius,radius:-radius]

    else:
        raise ValueError("unsupported filter method '{}'".format(method))

    return img_filtered


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage="%(prog)s [-h] [-m {naive,queue,opencv}] [-r RADIUS] {min,max} IMAGE",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('type',
                        choices=['min', 'max'],
                        help="filter type")

    parser.add_argument('image',
                        metavar='IMAGE',
                        help="input image")

    filter_modifiers_description = str(
        "-m, --method  {naive,queue,opencv}  filter implementation (default is naive)\n"
        "-r, --radius  RADIUS                filter radius (default is 1)")

    filter_modifiers_group = parser.add_argument_group(
        title="filter modifiers", description=filter_modifiers_description)

    filter_modifiers_group.add_argument('-m', '--method',
                                        default='naive',
                                        choices=['naive', 'queue', 'opencv'],
                                        help=argparse.SUPPRESS)

    filter_modifiers_group.add_argument('-r', '--radius',
                                        default=1,
                                        type=int,
                                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    img = cv.imread(args.image)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img = extremum_filter(img, args.type, radius=args.radius, method=args.method)

    cv.imshow("Result", img)

    cv.waitKey(0)
