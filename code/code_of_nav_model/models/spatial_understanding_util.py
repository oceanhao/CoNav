import torch
import numpy as np
from collections import defaultdict
import math


def get_angle_flags(angle_list=[]):
    angle_token_value = []
    for angle_iterm in angle_list:
        if angle_iterm == None:
            angle_token_value.append([0, 0, 0, 0, 0, 0])
        else:
            heading = angle_iterm["heading"]
            elevation = angle_iterm["elevation"]
            if -0.2 < heading < 0.2:
                heading = 0
            if -0.2 < elevation < 0.2:
                elevation = 0

            temp_list = get_direction_flags(heading)
            temp_list.append(0)
            temp_list.append(0)
            if -elevation > 0:
                temp_list[4] = 1
            if -elevation < 0:
                temp_list[5] = 1

            angle_token_value.append(temp_list)

    return angle_token_value


def normalize_angle(angle):

    while angle <= -math.pi:
        angle += 2 * math.pi
    while angle > math.pi:
        angle -= 2 * math.pi
    return angle


def get_direction_flags(a_heading_or_diff, b_heading=None):

    direction_flags = [0, 0, 0, 0]

    if b_heading == None:
        k = normalize_angle(a_heading_or_diff)
    else:
        k = normalize_angle(a_heading_or_diff - b_heading)

    if -math.pi / 2 <= k <= math.pi / 2:
        direction_flags[0] = 1
    else:
        direction_flags[1] = 1

    if k < -math.pi / 2 or k > math.pi / 2:

        if k > 0:
            direction_flags[3] = 1
        elif k < 0:
            direction_flags[2] = 1
    else:
        if k > 0:
            direction_flags[3] = 1
        elif k < 0:
            direction_flags[2] = 1

    return direction_flags
