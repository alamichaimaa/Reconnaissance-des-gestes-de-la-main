import copy

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage import io
from PIL import Image, ImageFilter
from scipy.spatial import distance
import math
from sympy import Point, Line, pi


def angles(image):

    # la couleur de chair
    """
    MIN = np.array([0, 42, 0], np.uint8)
    MAX = np.array([23, 222, 255], np.uint8)
    """


    #cv2.namedWindow("img3", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('img3', 700, 500)
    #cv2.imshow("img3", image)
    #cv2.waitKey(0)



    #partie labeling
    gray = image.copy()

    mask = np.where(image == 1, np.uint8(255), np.uint8(0))


    #io.imshow(img_label)
    #plt.show()

    #Optimizing the startpoint()
    def optimize(img,x,y,w,h):
        count=0
        for j in range(x,x+w-1,1):
            for i in range(y+h-1,y,-1):

                if img[i,j] == 255:
                    start_point = (i, j)
                    #print(start_point, img[i, j])
                    return start_point


    def first(img,x,y,w,h):
        count=0
        #print("the size of ",img.shape[0],img.shape[1])
        #print("the size of w,h",w,h)
        for i in range(0,w-1,1):
            for j in range(0,h-1,1):
                #print(j,i)

                if img[i,j] == 255:
                    start_point = (i, j)
                    #print(start_point, img[i, j])
                    return start_point



    #Code freeman for choosing hands
    def freeman(img, x, y,w,h):
        #h = filterImg.shape[0]
        #print('inside freeman:',img.shape[0])
        start_point = first(img,x,y,w,h)
        #print("this is a start point:",start_point)


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
        #print("this is the first borders", border)
        count = 0
        while curr_point != start_point:
            # figure direction to start search
            b_direction = (direction + 5) % 8
            dirs_1 = range(b_direction, 8)
            dirs_2 = range(0, b_direction)
            dirs = []
            dirs.extend(dirs_1)
            dirs.extend(dirs_2)
            #print(dirs)
            for direction in dirs:
                idx = dir2idx[direction]
                new_point = (curr_point[0] + change_i[idx], curr_point[1] + change_j[idx])
                # print("the coordonate",new_point,"the direction",direction,"the value is:",filterImg[new_point])

                if img_pil1[new_point] != 0 and new_point[0] < img.shape[0] - 1 and new_point[1] < img.shape[
                    1] - 1:  # if is ROI
                    # new_point=(new_point[0],abs(new_point[1]))

                    #print(new_point)
                    #print("deplacer de direction", direction)
                    border.append(new_point)
                    chain.append(direction)
                    curr_point = new_point
                    break
            if count == 10000: break
            count += 1

       # print(count)
        #print(chain)
        h = img_pil1.shape[0]
        return border, chain, start_point

    #plt.figure(4)

    img = copy.deepcopy(image)
    #plt.imshow(img, cmap='Greys')


    im_pil = Image.fromarray(img)
    img_pil1 = im_pil.filter(ImageFilter.ModeFilter(size=30))

    img_pil1 = np.array(img_pil1)
    #cv2.namedWindow("img_smoothed", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('img_smoothed', 700, 500)
    #cv2.imshow("img_smoothed", img_pil1)
    #cv2.waitKey(0)
    img = copy.deepcopy(img_pil1)
    # rotation dune image

    #img = copy.deepcopy(imgF)

    #cv2.imwrite("testharf2.jpeg",imgF)

    #img = copy.deepcopy(img_pil1)

    #plt.figure(4)
    #plt.imshow(img, cmap='Greys')

    c_area=1
    print(c_area,"this is c_area")

    for i in range(0, c_area, 1):
        #print(data[i])
        print(img.shape[0],img.shape[1])
        #print(d.shape[0])
        #res_mask = cv2.add(mask, img)
        border, chain, start_point = freeman(img, 0, 0, img.shape[0] ,img.shape[1])
        print("this is border")
        print(border)
        print(start_point)
        #rotate_img(border, chain, start_point)
       # plt.plot([i[1] for i in border], [i[0] for i in border])
       # plt.plot(start_point[1], start_point[0], marker=".", markersize=20)

    #plt.show()


    # essayons de detecter le poignet
    # traitement de corners


    def function1(chaine, border,r):
        # detection des changements brusque
        point_brusque = []
        k=0
        j=0

        while j < len(chaine):
           # print("this is the region number:",j, "and between", border[j])
            leng=r
            i=j
            pas=int(leng/18)
            maxi=0
            maxii=0
           # print("we bigin with between",i,i+leng)
            if i+leng>=len(chaine):
                break
            while i < j+leng :
                if i + pas >= len(chaine)-2:
                    break
                if i == len(chaine)-1:
                    break
                #print(i, i + pas)

                if abs(chaine[i] - chaine[i + pas]) >= maxii:
                   # print("the maximum index", i)
                    maxii = abs(chaine[i] - chaine[i + pas])
                    maxi = i
                i=i+pas
            point_brusque.append(border[int(maxi+pas/7)])
            j=j+r
        return point_brusque

    #plt.figure(4)
    #plt.imshow(img, cmap='Greys')

    #print("the length of border and chaine ",len(chain),len(border))

    point_brusque = function1(chain, border,int(len(chain)/5))

    #print('lenght of array:',len(point_brusque))
    #print(point_brusque)

    #plt.plot([i[1] for i in border], [i[0] for i in border])
    #plt.plot([i[1] for i in border], [i[0] for i in border], marker=".", markersize=2)
    #plt.plot(start_point[1], start_point[0], marker=".", markersize=20)

    #for i in range(0,len(point_brusque)-1,1):
        #print("Distance euclidien :", distance.euclidean(point_brusque[i], point_brusque[i+1]))

    #print("Distance euclidien :", distance.euclidean(point_brusque[0], point_brusque[len(point_brusque)-1]))

    #for i in range(0, len(point_brusque), 1):
        #plt.plot(point_brusque[i][1], point_brusque[i][0], marker=".", markersize=15)

    #plt.plot([i[1] for i in point_brusque], [i[0] for i in point_brusque])
    #plt.plot([point_brusque[0][1],point_brusque[len(point_brusque)-1][1]],[point_brusque[0][0],point_brusque[len(point_brusque)-1][0]])




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
    lineB = ((478, 480),(478, 378)) #DE
    lineA = ((478, 480),(114, 418))
    #print(ang(lineA, lineB))
    #Pour tout
    angels=[]
    for i in range(0, len(point_brusque)-2, 1):
        lineB = (point_brusque[i+1], point_brusque[i])  # DE
        lineA = (point_brusque[i+1], point_brusque[i+2])
        #plt.plot(point_brusque[i][1], point_brusque[i][0], marker=".", markersize=15)
        b=ang(lineA, lineB)
        #print("indexes:",i,i+1,i+2,b)
        angels.append(b)

    lineB = (point_brusque[len(point_brusque)-1], point_brusque[0])  # DE
    lineA = (point_brusque[len(point_brusque)-1], point_brusque[len(point_brusque)-2])
    #plt.plot(point_brusque[i][1], point_brusque[i][0], marker=".", markersize=15)
    b0=ang(lineA, lineB)
    #print("indexes:",i,i+1,i+2,b0)
    angels.append(b0)

    lineB = (point_brusque[0], point_brusque[len(point_brusque)-1])  # DE
    lineA = (point_brusque[0], point_brusque[1])
    #plt.plot(point_brusque[i][1], point_brusque[i][0], marker=".", markersize=15)
    bn = ang(lineA, lineB)
    #print("indexes:",i,i+1,i+2,bn)
    angels.append(bn)

   # plt.show()
    return angels

import glob
import csv 
file = "D:\\dataset\Data\Vao\*.jpg"
glob.glob(file)
with open('Data_alphabet.csv', 'a', newline='') as f_object:  

    for i in range(0,40,1):
        imagesAIN = [np.asarray(cv2.imread(image)) for image in glob.glob(file)]
        image1=imagesAIN[i]
        image= cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        img=copy.deepcopy(image)
        #calcule angles
        a=angles(img)
        lis=['Vao',a,19]

        # Pass the CSV  file object to the writer() function
        writer_object = csv.writer(f_object)
        writer_object.writerow(lis)  
# Close the file object
f_object.close()