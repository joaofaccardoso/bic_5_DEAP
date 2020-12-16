import sys
import math
from PIL import Image, ImageDraw
import numpy as np
import csv

# canonical interpolation function, like https://p5js.org/reference/#/p5/map
def map_number(n, start1, stop1, start2, stop2):
  return ((n-start1)/(stop1-start1))*(stop2-start2)+start2;

# input: array of real vectors, length 8, each component normalized 0-1
def render(a, size):
    # split input array into header and rest
    header_length = 2
    head = a[:header_length]
    rest = a[header_length:]

    # determine background color from header
    R = int(map_number(head[0][0], 0, 1, 0, 255))
    G = int(map_number(head[0][1], 0, 1, 0, 255))
    B = int(map_number(head[0][2], 0, 1, 0, 255))

    # create the image and drawing context
    im = Image.new('RGB', (size, size), (R, G, B))
    draw = ImageDraw.Draw(im, 'RGB')

    # now draw lines
    min_width = 0.004 * size
    max_width = 0.04 * size
    for e in rest:
        #print(len(e))
        x1 = map_number(e[0], 0, 1, 0, size)
        y1 = map_number(e[1], 0, 1, 0, size)
        x2 = map_number(e[2], 0, 1, 0, size)
        y2 = map_number(e[3], 0, 1, 0, size)
        x3 = map_number(e[4], 0, 1, 0, size)
        y3 = map_number(e[5], 0, 1, 0, size)


        # determine foreground color from header
        R = int(map_number(e[6], 0, 1, 0, 255))
        G = int(map_number(e[7], 0, 1, 0, 255))
        B = int(map_number(e[8], 0, 1, 0, 255))

        # draw line with round line caps (circles at the end)
        draw.polygon(
            ((x1,y1), (x2,y2), (x3,y3)), (R,G,B), outline=None)
    return im
