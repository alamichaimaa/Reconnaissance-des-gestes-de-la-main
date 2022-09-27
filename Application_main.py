import glob
import pandas as pd
import cv2
import cvzone
import numpy as np
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from numpy import arccos, rad2deg, sqrt
from sympy import false, true
import copy

import cv2
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage import io
from PIL import Image, ImageFilter
from scipy.spatial import distance
import math
from sympy import Point, Line, pi
import pickle
# take the picture
cam_port = 0
cam = cv2.VideoCapture(cam_port)

# reading the input using the camera
result, image = cam.read()
if result:
    cv2.imshow("GeeksForGeeks", image)

    # saving image in local storage
    cv2.imwrite("GeeksFor.jpg", image)
    cv2.waitKey(0)


# If captured image is corrupted, moving to else part
else:
    print("No image detected. Please! try again")


def pre_processing(image, left):
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # si c'est la main gauche en va inverser l'image
    if (left):
        image = cv2.flip(image, 2)
    # lisser l'image par filtre gaussien
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # supprimer l’arrière-plan
    segmentor = SelfiSegmentation()
    image = segmentor.removeBG(image, (0, 0, 0))
    cv2.namedWindow("img1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img1', 700, 500)
    cv2.imshow("img1", image)
    cv2.waitKey(0)
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # convertir l'image de l'espace BGR vers YCrImg et de de BGR ver HSV
    YCrImg = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    HSVImg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # fixer l'intervalle de couleur de la peau dans l'espace YCrCb
    MIN = np.array([0, 150, 100], np.uint8)
    MAX = np.array([250, 200, 150], np.uint8)
    # fixer l'intervalle de couleur de la peau dans l'espace HSV
    MIN1 = np.array([0, 30, 60], np.uint8)
    MAX1 = np.array([53, 255, 255], np.uint8)
    cv2.namedWindow("img2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img2', 700, 500)
    # filtrer l'image
    filterImg1 = cv2.inRange(YCrImg, MIN, MAX)
    filterImg2 = cv2.inRange(HSVImg, MIN1, MAX1)
    # combiner les resultat obtenu en utilisant le mask YCrCb et HSV
    filterImg = cv2.bitwise_or(filterImg1, filterImg2)
    cv2.imshow("img2", filterImg)
    cv2.waitKey(0)
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # h,w = filterImg.shape[:2]
    # mask= np.zeros((h+2,w+2),np.uint8)
    # imflood=filterImg.copy()
    # cv2.floodFill(imflood,mask,(0,0),255)
    # imfloodinv=cv2.bitwise_not(imflood)
    # filterImg=filterImg | imfloodinv
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # extraire l'element connex qui represente la main 
    ret, labels, stats = cv2.connectedComponentsWithStats(filterImg, 4)[0:3]
    minX = 100000
    imagF = np.zeros(filterImg.shape, np.uint8)
    rectangleH = []
    for label in range(1, ret):
        mask = np.array(labels, dtype=np.uint8)
        area = stats[label, cv2.CC_STAT_AREA]
        canvasH = np.zeros(filterImg.shape, np.uint8)
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w, h = stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
        if (area > 400):
            if (minX >= x):
                if (w > 220):
                    w = 220
                if (h > 250):
                    h = 250
                minX = x
                mask[labels == label] = 255
                imagF = mask.copy()
                cv2.rectangle(canvasH, (x, y), (x + w, y + h), (255, 255, 255), -1)
                imagF = cv2.bitwise_and(canvasH, imagF)
                rectangleH = [x, y, h, w]

    return imagF, rectangleH
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


imagetest = cv2.imread("GeeksFor.jpg")
imagetest, cordonnees = pre_processing(imagetest, true)
newimage = imagetest[cordonnees[1] - 20:cordonnees[1] + cordonnees[2], cordonnees[0] - 20:cordonnees[0] + cordonnees[3]]
print(cordonnees)
print(newimage.shape[0], newimage.shape[1])
cv2.namedWindow("img4", cv2.WINDOW_NORMAL)
cv2.resizeWindow('img4', 700, 500)
cv2.imshow("img4", newimage)
cv2.waitKey(0)


def angles(image):
    # la couleur de chair

    cv2.namedWindow("img3", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img3', 700, 500)
    cv2.imshow("img3", image)
    cv2.waitKey(0)

    # partie labeling
    gray = image.copy()

    mask = np.where(image == 1, np.uint8(255), np.uint8(0))

    # io.imshow(img_label)
    # plt.show()

    # Optimizing the startpoint()
    def optimize(img, x, y, w, h):
        count = 0
        for j in range(x, x + w - 1, 1):
            for i in range(y + h - 1, y, -1):

                if img[i, j] == 255:
                    start_point = (i, j)
                    # print(start_point, img[i, j])
                    return start_point

    def first(img, x, y, w, h):
        count = 0
        print("the size of ", img.shape[0], img.shape[1])
        print("the size of w,h", w, h)
        for i in range(0, w - 1, 1):
            for j in range(0, h - 1, 1):
                #print(j, i)

                if img[i, j] == 255:
                    start_point = (i, j)
                    # print(start_point, img[i, j])
                    return start_point

    # Code freeman for choosing hands
    def freeman(img, x, y, w, h):
        # h = filterImg.shape[0]
        print('inside freeman:', img.shape[0])
        start_point = first(img, x, y, w, h)
        print("this is a start point:", start_point)

        directions = [0, 1, 2,
                      7, 3,
                      6, 5, 4]
        dir2idx = dict(zip(directions, range(len(directions))))
        print("this is the dictionary:", dir2idx)
        change_j = [-1, 0, 1,  # x or columns
                    -1, 1,
                    -1, 0, 1]

        change_i = [-1, -1, -1,  # y or rows
                    0, 0,
                    1, 1, 1]

        border = []
        chain = []
        curr_point = start_point
        for direction in directions:
            idx = dir2idx[direction]
            new_point = (start_point[0] + change_i[idx], start_point[1] + change_j[idx])
            if img[new_point] != 0:  # if is ROI
                border.append(new_point)
                chain.append(direction)
                curr_point = new_point
                break
        # print("this is the first borders", border)
        count = 0
        while curr_point != start_point:
            # figure direction to start search
            b_direction = (direction + 5) % 8
            dirs_1 = range(b_direction, 8)
            dirs_2 = range(0, b_direction)
            dirs = []
            dirs.extend(dirs_1)
            dirs.extend(dirs_2)
            # print(dirs)
            for direction in dirs:
                idx = dir2idx[direction]
                new_point = (curr_point[0] + change_i[idx], curr_point[1] + change_j[idx])
                # print("the coordonate",new_point,"the direction",direction,"the value is:",filterImg[new_point])

                if img_pil1[new_point] != 0 and new_point[0] < img.shape[0] - 1 and new_point[1] < img.shape[
                    1] - 1:  # if is ROI

                    border.append(new_point)
                    chain.append(direction)
                    curr_point = new_point
                    break
            if count == 10000 : break
            count += 1
        #print(count)
        #print(chain)
        h = img_pil1.shape[0]
        return border, chain, start_point

    # plt.figure(4)

    img = copy.deepcopy(image)
    # plt.imshow(img, cmap='Greys')

    im_pil = Image.fromarray(img)
    img_pil1 = im_pil.filter(ImageFilter.ModeFilter(size=20))
    img_pil1 = np.array(img_pil1)
    img = copy.deepcopy(img_pil1)
    plt.figure(4)
    plt.imshow(img, cmap='Greys')
    c_area = 1
    print(c_area, "this is c_area")

    for i in range(0, c_area, 1):
        # print(data[i])
        print(img.shape[0], img.shape[1])
        border, chain, start_point = freeman(img, 0, 0, img.shape[0], img.shape[1])
        plt.plot([i[1] for i in border], [i[0] for i in border])
        plt.plot(start_point[1], start_point[0], marker=".", markersize=20)

    plt.show()

    # traitement de corners

    def function1(chaine, border, r):
        # detection des changements brusque
        point_brusque = []
        k = 0
        j = 0

        while j < len(chaine):
            # print("this is the region number:",j, "and between", border[j])
            leng = r
            i = j
            pas = int(leng / 18)
            maxi = 0
            maxii = 0
            # print("we bigin with between",i,i+leng)
            if i + leng >= len(chaine):
                break
            while i < j + leng:
                if i + pas >= len(chaine) - 2:
                    break
                if i == len(chaine) - 1:
                    break
                # print(i, i + pas)

                if abs(chaine[i] - chaine[i + pas]) >= maxii:
                    # print("the maximum index", i)
                    maxii = abs(chaine[i] - chaine[i + pas])
                    maxi = i
                i = i + pas
            point_brusque.append(border[int(maxi + pas / 7)])
            j = j + r
        return point_brusque

    plt.figure(4)
    plt.imshow(img, cmap='Greys')

    print("the length of border and chaine ", len(chain), len(border))

    point_brusque = function1(chain, border, int(len(chain) / 5))

    print('lenght of array:', len(point_brusque))
    print(point_brusque)

    plt.plot([i[1] for i in border], [i[0] for i in border])
    plt.plot([i[1] for i in border], [i[0] for i in border], marker=".", markersize=2)
    # plt.plot(start_point[1], start_point[0], marker=".", markersize=20)

    for i in range(0, len(point_brusque) - 1, 1):
        print("Distance euclidien :", distance.euclidean(point_brusque[i], point_brusque[i + 1]))

    print("Distance euclidien :", distance.euclidean(point_brusque[0], point_brusque[len(point_brusque) - 1]))

    for i in range(0, len(point_brusque), 1):
        plt.plot(point_brusque[i][1], point_brusque[i][0], marker=".", markersize=15)

    plt.plot([i[1] for i in point_brusque], [i[0] for i in point_brusque])
    plt.plot([point_brusque[0][1], point_brusque[len(point_brusque) - 1][1]],
             [point_brusque[0][0], point_brusque[len(point_brusque) - 1][0]])

    def dot(vA, vB):
        return vA[0] * vB[0] + vA[1] * vB[1]

    def ang(lineA, lineB):
        # Get nicer vector form
        vA = [(lineA[0][0] - lineA[1][0]), (lineA[0][1] - lineA[1][1])]
        vB = [(lineB[0][0] - lineB[1][0]), (lineB[0][1] - lineB[1][1])]
        # Get dot prod
        dot_prod = dot(vA, vB)
        # Get magnitudes
        magA = dot(vA, vA) ** 0.5
        magB = dot(vB, vB) ** 0.5
        # Get cosine value
        cos_ = dot_prod / magA / magB
        # Get angle in radians and then convert to degrees
        angle = math.acos(dot_prod / magB / magA)
        # Basically doing angle <- angle mod 360
        ang_deg = math.degrees(angle) % 360

        if ang_deg - 180 >= 0:
            # As in if statement
            return 360 - ang_deg
        else:

            return ang_deg

    lineB = ((478, 480), (478, 378))  # DE
    lineA = ((478, 480), (114, 418))
    print(ang(lineA, lineB))
    # Pour tout
    angels = []
    for i in range(0, len(point_brusque) - 2, 1):
        lineB = (point_brusque[i + 1], point_brusque[i])  # DE
        lineA = (point_brusque[i + 1], point_brusque[i + 2])
        # plt.plot(point_brusque[i][1], point_brusque[i][0], marker=".", markersize=15)
        b = ang(lineA, lineB)
        print("indexes:", i, i + 1, i + 2, b)
        angels.append(b)

    lineB = (point_brusque[len(point_brusque) - 1], point_brusque[0])  # DE
    lineA = (point_brusque[len(point_brusque) - 1], point_brusque[len(point_brusque) - 2])
    # plt.plot(point_brusque[i][1], point_brusque[i][0], marker=".", markersize=15)
    b0 = ang(lineA, lineB)
    print("indexes:", i, i + 1, i + 2, b0)
    angels.append(b0)

    lineB = (point_brusque[0], point_brusque[len(point_brusque) - 1])  # DE
    lineA = (point_brusque[0], point_brusque[1])
    # plt.plot(point_brusque[i][1], point_brusque[i][0], marker=".", markersize=15)
    bn = ang(lineA, lineB)
    print("indexes:", i, i + 1, i + 2, bn)
    angels.append(bn)

    plt.show()
    return angels


# debut de fonction
# image1 = cv2.imread("harf/imagemeem17.jpg")
image1 = copy.deepcopy(newimage)
print(image1.dtype)
# image= cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
img = copy.deepcopy(image1)
a = angles(img)

while( len(a) <5):
    a.append(180)
for i in range(len(a)):
    a[i]=a[i]*1/100
print(a)

data=pd.read_csv("Data_alphabet.csv")
def nomclasse(ypredct):
    j=0
    A=data.iloc[:,0]
    if (ypredct==j):
        return A[j]
    if(ypredct==1):
        j=j+1
        return A[j*40]
    if(ypredct<15):
        while(j!= ypredct):
            j=j+1
        return A[j*39]
    else:
        while(j!= ypredct):
            j=j+1
        return A[j*37]
# Testing sur un model
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
result = loaded_model.predict([a])
print('the resulte of the model is :',nomclasse(result))
print("expected :", 15)