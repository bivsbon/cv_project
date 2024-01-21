from concurrent.futures import ThreadPoolExecutor, wait

import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from math import atan, pi


def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def seg_intersect(a1, a2, b1, b2):
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return ((num / denom) * db + b1).astype(int).tolist()


HORIZONTAL = 0
VERTICAL = 1
M_X = 650
M_Y = 1100

TOP_V_LINE_RATIO = 10
BOT_V_LINE_RATIO = 20
TOP_H_LINE_RATIO = 15
BOT_H_LINE_RATIO = 60

BASE_POINT_1 = (1120, 200)
BASE_POINT_2 = (1128, 2161)

NAME1_RECT_BASE_1 = (304, 58)
NAME1_RECT_BASE_2 = (1048, 147)

DOB1_RECT_BASE_1 = (455, 147)
DOB1_RECT_BASE_2 = (887, 209)

NAME2_RECT_BASE_1 = (266, 477)
NAME2_RECT_BASE_2 = (1013, 567)

DOB2_RECT_BASE_1 = (464, 567)
DOB2_RECT_BASE_2 = (911, 628)


def near(x, y, distance=30):
    return abs(x - y) < distance


def is_diagonal_line(x1, y1, x2, y2, alpha=pi/20):
    if x1 == x2:
        return False

    angle = atan(abs(y1-y2) / abs(x1-x2))
    return alpha < angle < pi/2 - alpha


def group_line(x1, y1, x2, y2, x_clusters, y_clusters):
    # Eliminate diagonal lines
    if not is_diagonal_line(x1, y1, x2, y2):
        if abs(x1 - x2) < abs(y1 - y2):
            added = False
            for cluster in x_clusters:
                for point in cluster:
                    x, _ = point
                    if near(x, x1) or near(x, x2):
                        cluster.append((x1, y1))
                        cluster.append((x2, y2))
                        added = True
                        break
                if added:
                    break
            if not added:
                x_clusters.append([(x1, y1), (x2, y2)])
        else:
            added = False
            for cluster in y_clusters:
                for point in cluster:
                    _, y = point
                    if near(y, y1) or near(y, y2):
                        cluster.append((x1, y1))
                        cluster.append((x2, y2))
                        added = True
                        break
                if added:
                    break
            if not added:
                y_clusters.append([(x1, y1), (x2, y2)])
        return True
    else:
        return False


def merge_lines(x_clusters, y_clusters, min_length=450):
    h_lines = []
    v_lines = []
    for cluster in x_clusters:
        low_endpoint = max(cluster, key=lambda point: point[1])
        high_endpoint = min(cluster, key=lambda point: point[1])
        if manhattan(low_endpoint, high_endpoint) > min_length:
            v_lines.append((low_endpoint, high_endpoint))
    for cluster in y_clusters:
        right_endpoint = max(cluster, key=lambda point: point[0])
        left_endpoint = min(cluster, key=lambda point: point[0])
        if manhattan(left_endpoint, right_endpoint) > min_length:
            h_lines.append((right_endpoint, left_endpoint))
    return v_lines, h_lines


def compute_intersections(v_lines, h_lines):
    intersections = []
    for v_line in v_lines:
        for h_line in h_lines:
            intersections.append(
                seg_intersect(np.array(v_line[0]), np.array(v_line[1]), np.array(h_line[0]), np.array(h_line[1])))
    return intersections


def group_into_quadrants(points):
    top_left_quad = []
    bottom_left_quad = []
    top_right_quad = []
    bottom_right_quad = []
    for point in points:
        x = point[0]
        y = point[1]
        if x < M_X and y < M_Y:
            top_left_quad.append((x, y))
        elif x < M_X and y > M_Y:
            bottom_left_quad.append((x, y))
        elif x > M_X and y < M_Y:
            top_right_quad.append((x, y))
        else:
            bottom_right_quad.append((x, y))
    return top_left_quad, bottom_left_quad, top_right_quad, bottom_right_quad


def filter_clusters(clusters, min_size=2):
    return [cluster for cluster in clusters if len(cluster) >= min_size]


def process(in_path):
    img = cv.imread(in_path, cv2.IMREAD_GRAYSCALE)[:, 0:-100]
    (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Edge detection
    dst = cv.Canny(im_bw, 50, 200, None, 5)

    # Line detection
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 48, 1)

    # Lines with roughly equal x and y are grouped together
    x_clusters = []
    y_clusters = []
    if linesP is not None:
        for lineP in linesP:
            l = lineP[0]
            x1, y1, x2, y2 = l[0], l[1], l[2], l[3]
            # print('({}, {}) -> ({}, {})'.format(x1, y1, x2, y2))
            if group_line(x1, y1, x2, y2, x_clusters, y_clusters):
                # cv.line(img, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 7, cv.LINE_AA)
                pass

    # Drop clusters with too small size
    x_clusters = filter_clusters(x_clusters, min_size=4)
    y_clusters = filter_clusters(y_clusters, min_size=4)

    # Merge small lines
    correct = True
    # for cluster in x_clusters:
    #     print('x', len(cluster))
    # for cluster in y_clusters:
    #     print('y', len(cluster))
    v_lines, h_lines = merge_lines(x_clusters, y_clusters)
    # print(v_lines)
    # print(h_lines)
    for line in v_lines:
        (x1, y1), (x2, y2) = line
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 255), 3, cv.LINE_AA)
    for line in h_lines:
        (x1, y1), (x2, y2) = line
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 255), 3, cv.LINE_AA)

    intersections = compute_intersections(v_lines, h_lines)

    # Group corners into 4 quadrants
    top_left_quad, bottom_left_quad, top_right_quad, bottom_right_quad = group_into_quadrants(intersections)

    # Find the closest point to the center of each quadrant
    top_left_point = min(top_left_quad, key=lambda p: abs(p[0] - M_X) + abs(p[1] - M_Y))
    top_right_point = min(top_right_quad, key=lambda p: abs(p[0] - M_X) + abs(p[1] - M_Y))
    bottom_left_point = min(bottom_left_quad, key=lambda p: abs(p[0] - M_X) + abs(p[1] - M_Y))
    bottom_right_point = min(bottom_right_quad, key=lambda p: abs(p[0] - M_X) + abs(p[1] - M_Y))

    # Automatically evaluate these 4 points , they should somewhat form a rectangle
    filter_dist = 70
    if not near(top_left_point[0], bottom_left_point[0], filter_dist) or \
            not near(top_left_point[1], top_right_point[1], filter_dist) or \
            not near(bottom_right_point[0], top_right_point[0], filter_dist) or \
            not near(bottom_right_point[1], bottom_left_point[1], filter_dist):
        correct = False

    # Circle these points
    img = cv2.circle(img, center=top_left_point, radius=10, color=(0, 0, 255), thickness=-1)
    img = cv2.circle(img, center=top_right_point, radius=10, color=(0, 0, 255), thickness=-1)
    img = cv2.circle(img, center=bottom_left_point, radius=10, color=(0, 0, 255), thickness=-1)
    img = cv2.circle(img, center=bottom_right_point, radius=10, color=(0, 0, 255), thickness=-1)

    pts1 = np.float32([top_left_point, top_right_point, bottom_left_point, bottom_right_point])
    pts2 = np.float32([[0, 0], [1070, 0], [0, 1960], [1070, 1960]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    img = cv.warpPerspective(img, M, (1070, 1960))

    cv.rectangle(img, NAME1_RECT_BASE_1, NAME1_RECT_BASE_2, (0, 0, 255), thickness=6)
    cv.rectangle(img, DOB1_RECT_BASE_1, DOB1_RECT_BASE_2, (0, 0, 255), thickness=6)
    cv.rectangle(img, NAME2_RECT_BASE_1, NAME2_RECT_BASE_2, (0, 0, 255), thickness=6)
    cv.rectangle(img, DOB2_RECT_BASE_1, DOB2_RECT_BASE_2, (0, 0, 255), thickness=6)

    return img, correct


def process_and_save(in_path):
    img, correct = process(in_path)
    # plt.figure()
    # plt.imshow(img)
    # plt.show()

    if correct:
        cv.imwrite('out/correct/{}.jpg'.format(i + 1), img)
    else:
        cv.imwrite('out/incorrect/{}.jpg'.format(i + 1), img)
    print(in_path)


def process_and_show(in_path):
    img, correct = process(in_path)

    plt.figure()
    plt.imshow(img)
    plt.show()


executor = ThreadPoolExecutor(max_workers=6)
futures = []
for i in range(200):
    futures.append(executor.submit(process_and_save, 'img/{}.jpg'.format(i + 1)))
wait(futures)
print('Done')
#
# process_and_show("img/15.jpg")
