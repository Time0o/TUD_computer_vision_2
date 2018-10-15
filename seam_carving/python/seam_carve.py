#!/usr/bin/env python3

import argparse
import os
import sys

DESC = """\
An implementation of the seam carving image retargeting algorithm.
Use either --width or --height to specify the dimensions of the output image."""


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(
        description=DESC, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--width',
                        type=int,
                        help="width of output image")

    parser.add_argument('--height',
                        type=int,
                        help="height of output image")

    parser.add_argument('--line-aware',
                        action='store_true',
                        help="try to preserve straight lines")

    parser.add_argument('-o', '--out',
                        help="write the result to OUTFILE")

    parser.add_argument('-i', '--in-place',
                        action='store_true',
                        help="replace the input image with the result")

    parser.add_argument('-v', '--visualize',
                        action='store_true',
                        help="visualize the result")

    args = parser.parse_args()

    err_prefix = "{}: error: ".format(os.path.basename(sys.argv[0]))

    if args.width is None and args.height is None:
        print(err_prefix + "either --width or --height must be specified")
        sys.exit(1)

    if args.width is not None and args.height is not None:
        print(err_prefix + "--width and --height may not be specified together")
        sys.exit(1)
