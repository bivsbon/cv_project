from concurrent.futures import ThreadPoolExecutor, wait

import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from math import atan, pi, sqrt
from constants import *


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


def near(point, line, distance=10):
    (x1, y1), (x2, y2) = line
    (a, b) = point
    d = abs((x2-x1) * (b-y1) - (y2-y1) * (a-x1)) / sqrt((x2-x1)**2 + (y2-y1)**2)
    # if d < 50:
    #     print("Line: ({}, {}) -> ({}, {})".format(x1, y1, x2, y2))
    #     print("Point: ({}, {})".format(a, b))
    #     print("Distance: ", d)
    #     print()
    return d < distance


def is_horizontal(line, alpha=pi/20):
    (x1, y1), (x2, y2) = line
    if x1 == x2:
        return False

    angle = atan(abs(y1 - y2) / abs(x1 - x2))
    return angle < alpha


def is_vertical(line, alpha=pi/20):
    (x1, y1), (x2, y2) = line
    if x1 == x2:
        return True

    angle = atan(abs(y1 - y2) / abs(x1 - x2))
    return angle > pi / 2 - alpha


def group(lines):
    clusters = []
    for line in lines:
        added = False
        p1, p2 = line
        for cluster in clusters:
            for cluster_line in cluster:
                if near(p1, cluster_line) and near(p2, cluster_line):
                    cluster.append(line)
                    added = True
                    break
            if added:
                break
        if not added:
            clusters.append([line])
    return clusters


def merge_lines(horizontal_line_clusters, vertical_line_clusters, min_length=450):
    horizontal_lines = []
    vertical_lines = []
    for cluster in vertical_line_clusters:
        low_endpoint = max(cluster, key=lambda line: line[1][1])[1]
        high_endpoint = min(cluster, key=lambda line: line[0][1])[0]
        if manhattan(low_endpoint, high_endpoint) > min_length:
            vertical_lines.append((low_endpoint, high_endpoint))
    for cluster in horizontal_line_clusters:
        right_endpoint = max(cluster, key=lambda line: line[1][0])[1]
        left_endpoint = min(cluster, key=lambda line: line[0][0])[0]
        if manhattan(left_endpoint, right_endpoint) > min_length:
            horizontal_lines.append((right_endpoint, left_endpoint))
    return horizontal_lines, vertical_lines


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
    (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Edge detection
    dst = cv.Canny(img, 50, 200, None, 5)

    # Line detection
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 150, None, 48, 1)

    # Lines with roughly equal x and y are grouped together
    vertical_lines = []
    horizontal_lines = []

    # Remove diagonal lines, only keep vertical and horizontal lines
    if linesP is not None:
        for lineP in linesP:
            l = lineP[0]
            x1, y1, x2, y2 = l[0], l[1], l[2], l[3]
            cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 7, cv.LINE_AA)
            line = ((x1, y1), (x2, y2))
            if is_vertical(line):
                vertical_lines.append(line)
            elif is_horizontal(line):
                horizontal_lines.append(line)

    # Group together lines that are almost identical to each other
    horizontal_line_clusters = group(horizontal_lines)
    vertical_line_clusters = group(vertical_lines)

    # Drop clusters with too small size
    horizontal_line_clusters = filter_clusters(horizontal_line_clusters, min_size=4)
    vertical_line_clusters = filter_clusters(vertical_line_clusters, min_size=4)

    # Merge lines in a cluster into one long line, that should be the edges of the table in the document
    correct = True
    for cluster in horizontal_line_clusters:
        print('x', len(cluster))
    for cluster in vertical_line_clusters:
        print('y', len(cluster))
    h_lines, v_lines = merge_lines(horizontal_line_clusters, vertical_line_clusters)
    print(v_lines)
    print(h_lines)
    for line in v_lines:
        (x1, y1), (x2, y2) = line
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 255), 10, cv.LINE_AA)
    for line in h_lines:
        (x1, y1), (x2, y2) = line
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 255), 10, cv.LINE_AA)

    intersections = compute_intersections(v_lines, h_lines)

    # Group corners into 4 quadrants accroding to middle point
    top_left_quad, bottom_left_quad, top_right_quad, bottom_right_quad = group_into_quadrants(intersections)

    # Find the closest point to the center of each quadrant
    top_left_point = min(top_left_quad, key=lambda p: abs(p[0] - M_X) + abs(p[1] - M_Y))
    top_right_point = min(top_right_quad, key=lambda p: abs(p[0] - M_X) + abs(p[1] - M_Y))
    bottom_left_point = min(bottom_left_quad, key=lambda p: abs(p[0] - M_X) + abs(p[1] - M_Y))
    bottom_right_point = min(bottom_right_quad, key=lambda p: abs(p[0] - M_X) + abs(p[1] - M_Y))

    # (deprecated) Automatically evaluate these 4 points, they should somewhat form a rectangle
    # filter_dist = 70
    # if not near(top_left_point[0], bottom_left_point[0], filter_dist) or \
    #         not near(top_left_point[1], top_right_point[1], filter_dist) or \
    #         not near(bottom_right_point[0], top_right_point[0], filter_dist) or \
    #         not near(bottom_right_point[1], bottom_left_point[1], filter_dist):
    #     correct = False

    # Circle these points
    # img = cv2.circle(img, center=top_left_point, radius=10, color=(0, 0, 255), thickness=-1)
    # img = cv2.circle(img, center=top_right_point, radius=10, color=(0, 0, 255), thickness=-1)
    # img = cv2.circle(img, center=bottom_left_point, radius=10, color=(0, 0, 255), thickness=-1)
    # img = cv2.circle(img, center=bottom_right_point, radius=10, color=(0, 0, 255), thickness=-1)

    # Warp four corners into rectangle than get 4 bounding box
    pts1 = np.float32([top_left_point, top_right_point, bottom_left_point, bottom_right_point])
    pts2 = np.float32([[0, 0], [1070, 0], [0, 1960], [1070, 1960]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    img = cv.warpPerspective(img, M, (1070, 1960))

    # Draw bounding boxes, and we're done
    cv.rectangle(img, NAME1_RECT_BASE_1, NAME1_RECT_BASE_2, (0, 0, 255), thickness=6)
    cv.rectangle(img, DOB1_RECT_BASE_1, DOB1_RECT_BASE_2, (0, 0, 255), thickness=6)
    cv.rectangle(img, NAME2_RECT_BASE_1, NAME2_RECT_BASE_2, (0, 0, 255), thickness=6)
    cv.rectangle(img, DOB2_RECT_BASE_1, DOB2_RECT_BASE_2, (0, 0, 255), thickness=6)

    return img, correct


def process_and_save(in_img, out_img):
    img, correct = process("img/{}".format(in_img))
    if correct:
        cv.imwrite('out/correct/{}'.format(out_img), img)
    else:
        cv.imwrite('out/incorrect/{}'.format(out_img), img)


def process_and_save_all():
    executor = ThreadPoolExecutor(max_workers=6)
    futures = []
    for i in range(200):
        futures.append(executor.submit(process_and_save, '{}.jpg'.format(i + 1), '{}.jpg'.format(i + 1)))
    wait(futures)
    print('Done')
    

def process_and_show(in_path):
    img, correct = process(in_path)

    plt.figure()
    plt.imshow(img)
    plt.show()


# process_and_save_all()
process_and_show("img/179.jpg")
