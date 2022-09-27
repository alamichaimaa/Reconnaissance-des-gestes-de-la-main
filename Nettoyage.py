import cv2
from PIL import Image,ImageFilter
import numpy as np
import glob
file ="D:\\dataset\PSL\\Ain\*.png"

glob.glob(file)

imagesAIN = [np.asarray(cv2.imread(image)) for image in glob.glob(file)]

for i in range(40):
    imagetest=imagesAIN[i]

    cv2.imshow("img2",imagetest)
    cv2.waitKey(0)
    #********* convertir image en niveau de gris 
    imagetest1 = cv2.GaussianBlur(imagetest,(5,5),0) 
    image = cv2.cvtColor(imagetest1, cv2.COLOR_BGR2GRAY )
    cv2.imshow("gray",image)
    cv2.waitKey(0)
    
    #_,thr=cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
    #cv2.imshow("binaire",thr)
    #cv2.waitKey(0)
    
    #******** filtre median pour reduire le bruit
    img=cv2.medianBlur(image,5)
    
    #cv2.imshow("madian",img)
    #cv2.waitKey(0)
    #Rechape image
    pixel_vals = img.reshape((-1,3)) # numpy reshape operation -1 unspecified 
    
    #************ Kmeans
    # Convert to float type only for supporting cv2.kmean
    pixel_vals = np.float32(pixel_vals)
    #criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    # Choosing number of cluster
    k = 2
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 
    # convert data into 8-bit values 
    centers = np.uint8(centers) 
    segmented_data = centers[labels.flatten()] # Mapping labels to center points( RGB Value)
    # reshape data into the original image dimensions 
    segmented_image = segmented_data.reshape((img.shape)) 
    #cv2.imshow("kmeans",segmented_image)
    #cv2.waitKey(0)
    ####################### convertir image en binaire
    _,thr1=cv2.threshold(segmented_image,127,255,cv2.THRESH_BINARY_INV)
    cv2.imshow("binaire",thr1)
    cv2.waitKey(0)
    
    #*************** Remplir 
    h,w = thr1.shape[:2]
    mask=np.zeros((h+2,w+2),np.uint8)
    imflood=thr1.copy()
    cv2.floodFill(imflood,mask,(0,0),255)
    imfloodinv=cv2.bitwise_not(imflood)
    thr1=thr1 | imfloodinv
    
	#******************  Extraire l’élément connexe 
    ret,labels, stats = cv2.connectedComponentsWithStats(thr1,4)[0:3]
    minX=100000
    imagF= np.zeros(thr1.shape,np.uint8)
    rectangleH=[]
    for label in range(1,ret):
         mask = np.array(labels, dtype=np.uint8)
         area=stats[label, cv2.CC_STAT_AREA]
         canvasH= np.zeros(thr1.shape,np.uint8)
         x=stats[label, cv2.CC_STAT_LEFT]
         y=stats[label, cv2.CC_STAT_TOP]
         w, h = stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
         if(area>400):
             if(minX>=x):
                 if(w>600):
                    w=600   
                 if(h>400):
                    h=400
                 minX=x
                 mask[labels == label] = 255
                 imagF=mask.copy()
                 cv2.rectangle(canvasH,(x,y),(x+w,y+h),(255,255,255),-1)
                 imagF=cv2.bitwise_and(canvasH,imagF)
                 rectangleH=[x,y,h,w]

    #*********************** lissage
    im_pil = Image.fromarray(imagF)
    img_pil1 = im_pil.filter(ImageFilter.ModeFilter(size=30))
    img_pil1 = np.array(img_pil1)
    
   # cv2.imwrite(path, img_pil1)
    #cv2.waitKey(0)
    cv2.imshow(" image m9at3a", imagF)
    cv2.waitKey(0)
    #************* enregistrer image en un path 
    path="D:\\dataset\Data\Ain\image"+str(i)+".jpg"

    cv2.imwrite(path, img_pil1)
    cv2.waitKey(0)