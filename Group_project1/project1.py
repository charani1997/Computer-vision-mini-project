import cv2
import numpy as np
import matplotlib.pyplot as plt


def extractSkin(img):
    img1=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #converting from gbr to hsv color space
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #skin color range for hsv color space 
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,160,255)) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #skin color range for hsv color space 
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #merge skin detection (YCbCr and hsv)
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))


    HSV_result = cv2.bitwise_not(HSV_mask)
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)
    global_result=cv2.bitwise_not(global_mask)

    ret,thresh=cv2.threshold(global_result,205,255,cv2.THRESH_BINARY_INV)

    ##mask
    ##mask=cv2.inRange(thresh,(0,0,255),(0,255,255))

    ##applying a mask to the image
    result=cv2.bitwise_and(img1,img1,mask=global_mask)
    return result

#Open a simple image
img1=cv2.imread(r"D:\400L\CS 408 COMPUTER VISION\Project\1_Group project1- Devise a simple skin colour detector\sample img\1.jpg", cv2.IMREAD_COLOR)
img2=cv2.imread(r"D:\400L\CS 408 COMPUTER VISION\Project\1_Group project1- Devise a simple skin colour detector\sample img\2.jpg", cv2.IMREAD_COLOR)
img3=cv2.imread(r"D:\400L\CS 408 COMPUTER VISION\Project\1_Group project1- Devise a simple skin colour detector\sample img\3.jpg", cv2.IMREAD_COLOR)
img4=cv2.imread(r"D:\400L\CS 408 COMPUTER VISION\Project\1_Group project1- Devise a simple skin colour detector\sample img\10.jpg", cv2.IMREAD_COLOR)



##data=[img1, img2, img3, img4, img5, img6]
data=[img1,img2,img3,img4]
final=[]

for i in range(0,len(data)):
    final.append(extractSkin(data[i]))

k=1
for i in range(0,len(data)):
    plt.subplot(len(data),2,k)
    plt.title('Original')
    plt.imshow(cv2.cvtColor(data[i], cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])

    plt.subplot(len(data),2,(k+1))
    plt.title('Detected Skin')
    plt.imshow(final[i])
    plt.xticks([])
    plt.yticks([])

    k+=2
  

##plt.subplot(1,2,1)
##plt.title('Original')
##plt.imshow(final[0])
##plt.xticks([])
##plt.yticks([])

##plt.subplot(1,2,2)
##plt.title('HSV_result')
##plt.imshow(HSV_result,cmap="Greys_r")
##plt.xticks([])
##plt.yticks([])
##
##plt.subplot(1,2,3)
##plt.title('YCrCb_result')
##plt.imshow(YCrCb_result,cmap="Greys_r")
##plt.xticks([])
##plt.yticks([])

##plt.subplot(1,2,2)
##plt.title('global_result')
##plt.imshow(final[1])
##plt.xticks([])
##plt.yticks([])

plt.show()
