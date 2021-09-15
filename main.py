
"""
 COLOUR CODING
-----------------

white - 1
red - 2
orange - 3
yellow - 4
green - 5
blue - 6
"""

import os
import cv2 as cv
import numpy as np


def colour_finder(list):
    cc = 0
    [H, S, V] = list
    if S <= 50 and V >= 160:
        cc = 1
    elif H > 6 and H <= 23:
        cc = 3
    elif H > 23 and H <= 50:
        cc = 4
    elif H > 50 and H <= 100:
        if S > 170:
            cc = 5
        else:
            cc = 1
    elif H > 100 and H <= 160:
        if S > 170:
            cc = 6
        else:
            cc = 1
    else:
        cc = 2

    return cc


def theta_rot(edge, mid):
    [x1, y1] = edge
    [x0, y0] = mid
    slope = -(y1 - y0) / (x1 - x0)
    theta = (np.arctan(slope)) * (57.29577951)
    if theta < 0:
        theta = theta + 180
    if slope >= 0:
        phi = 45 - theta
    else:
        phi = 135 - theta

    return phi


path = "input"
temp = os.listdir(path)

for path_image in range(len(temp)) :
    image = cv.imread(temp[path_image])
    p_img = temp[path_image].split(".")
    area = []
    height = []
    width = []
    (h1, w1, c1) = image.shape
    #print(image.shape)

    img = image.copy()
    blank_image = np.zeros((image.shape[0], image.shape[1], 3))
    image_lab = cv.cvtColor(image,cv.COLOR_BGR2LAB)
    image_hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)

    g_blur = cv.GaussianBlur(image_lab, (5, 5), 0)
    image_canny = cv.Canny(g_blur, 50, 150)

    contours, hierarchy = cv.findContours(image_canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(img, contours, 0, color = (0, 255, 0), thickness=1)
    sorted_contours = sorted(contours, key=cv.contourArea)
    for contour in sorted_contours :
        x, y, w, h = cv.boundingRect(contour)
        dst = abs(h-w)
        if dst <= 5 and dst >=0 and h*w > 0.005*h1*w1 :
            h_i, s, v = cv.split(image_hsv[y + (h // 3):y + (2 * h // 3), x + w // 3:x + (2 * w // 3)])
            Hi = int(np.mean(h_i))
            l_Hi = len(h_i)
            h_var = ((h_i - Hi) ** 2) // l_Hi
            h_var_mean = np.mean(h_var)
            if h_var_mean <= 0.00001 :
                area.append([x, y, w, h])

    bbox = []
    [bbox.append([x, y, w, h]) for [x, y, w, h] in area if [x, y, w, h] not in bbox]

    n = len(bbox)//2
    [x_ref, y_ref, w_ref, h_ref] = bbox[n]
    center = []
    count = 0
    for i in range(len(bbox)) :
        [x,y,w,h] = bbox[i]
        h_mod = abs(h - h_ref)
        w_mod = abs(w - w_ref)
        if h_mod<=4 and w_mod<=4 :
            cv.rectangle(img, (x, y), (x + w, y + h), color=(50, 120, 250), thickness=2)
            cv.rectangle(blank_image, (x, y), (x + h, y + w), color=(200, 120, 56), thickness=2)
            count = count +1
            h_i, s, v = cv.split(image_hsv[y+(h//3):y+(2*h//3), x+w//3:x+(2*w//3)])
            Hi = int(np.mean(h_i))
            Si = int(np.mean(s))
            Vi = int(np.mean(v))
            color = [Hi, Si, Vi]
            c_code = colour_finder(color)
            cx = (2*x + w)//2
            cy = (2*y + w)//2
            cv.putText(img, str(c_code), (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness = 1)
            center.append([cx, cy, c_code])


    center.sort(key=lambda x: x[1])
    [cx1, cy1, cc1] = center[0]
    [cx0, cy0, cc2] = np.mean(center, axis=0).astype(int)

    phi = theta_rot([cx1, cy1], [cx0, cy0])

    rot_matrix = cv.getRotationMatrix2D((int(cx0), int(cy0)), phi, scale=1)
    rot_image = cv.warpAffine(img, rot_matrix, dsize=(w1, h1))
    phi_rad = phi/57.29577951

    for i in range(len(center)) :
        [cx1, cy1, p] = center[i]
        cx1 = cx1 - cx0
        cy1 = cy1- cy0
        cy_new = int(cy1*np.cos(phi_rad) - cx1*np.sin(phi_rad)) + cy0
        cx_new = int(cx1*np.cos(phi_rad) + cy1*np.sin(phi_rad)) +cx0
        center[i] = [cx_new, cy_new, p]

    c =[]
    center.sort(key= lambda x:x[1])
    c.append(sorted(center[0:3], key=lambda x:x[0]))
    c.append(sorted(center[3:6], key=lambda x:x[0]))
    c.append(sorted(center[6:9], key=lambda x:x[0]))

    final_matrix = []
    for i in range(3) :
        for j in range(3) :
            final_matrix.append(c[i][j][2])

    final_matrix = [str(final_matrix[0]),"\t\t", str(final_matrix[1]),"\t\t", str(final_matrix[2]), "\n\n", str(final_matrix[3]),"\t\t",
                    str(final_matrix[4]),"\t\t", str(final_matrix[5]), "\n\n", str(final_matrix[6]),"\t\t", str(final_matrix[7]),"\t\t", str(final_matrix[8]), "\n"]


    # add a function where you can check if a square is detected or not and thus draw a bbox around it
    #cv.imshow("img", img)
    #cv.imshow("rotated image", rot_image)
    #cv.imshow("image", image_canny)
    #cv.imshow("contours", blank_image)
    #cv.imshow("ba", color_img)
    #cv.waitKey(1)
    file = open("Output/Output_" + str(p_img[0]) + ".txt", "w")
    file.writelines(final_matrix)
    file.close()

