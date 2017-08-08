#encoding=utf-8
import os
import cv2
import numpy as np
from numpy import dtype
import logging
import math
from copy import copy
from cv2 import waitKey
def contourArea(img,contour):
    r = 0
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            # print(x,y)
            r += 1 if cv2.pointPolygonTest(contour,(x,y),False) > 0 else 0
    return r
# def tryFindMinAreaRect(img_list):
#     r = []
#     try:
#         img = cv2.imread("/tmp/armor_detect_tmp.png")
#         img_back = copy(img)
# #         img_resize = cv2.resize(img,(64, 32), interpolation = cv2.INTER_CUBIC)
#         gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)        
#         gray_gaussion = cv2.GaussianBlur(gray, (1,1), 305)
#         ret,binary = cv2.threshold(gray_gaussion,np.mean(img_back),255,cv2.THRESH_BINARY) 
#         cannys = cv2.Canny(binary,0,180)
#         b,contours,h = cv2.findContours(cannys,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_NONE);   
#         for c in contours:
#             if len(c) > 45  and len(c) < 80 and contourArea(img_back,c) > 170 and contourArea(img_back,c)/len(c) > 3.5:
#                 img2 = copy(img_back)
#                 cv2.drawContours(img2,c,-1,(0,0,255),1)
#                 cv2.imshow("img2",img2)  
#                 x,y,w,h = cv2.boundingRect(c)
#                 cv2.circle(img2,(x,y), 16, (0,255,127), 1)                         
#                 r.append((1,x,y,w,h))
#         if len(r) > 1:
#             logging.warn("[Warn],found too many circles...")
#             return r[0]
#         elif len(r) == 1:
#             print('[info],',r[0])
#             return r[0]
#         else:
#             return (0,0,0,0,0)
#     except Exception as e:
#         logging.exception(e)

if __name__ == "__main__":
    filesdir = "C://Users/kohillyang/OneDrive/2017/cnn/cv2Test/train/0"
    tryFindMinAreaRect(None)    
    print("[info]","hello world")
def xy2rad(x,y,x0=16,y0=16):
    return np.atan2(x,y) + 3.14,np.sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0))
def rad2xy(d,r,x0=16,y0=16):
    print d
    x = r * int(float(math.cos(d/6.29)))
    y = r * int(float(math.sin(d/6.29)))
    print x,y,r,d
    return int(float(x)),int(float(y))
def points(ra):
    r = []
    count = int(16*ra[0])
    for i in range(count):
        print count
        rxy = rad2xy(6.28/count*i,ra)
        r.append(rxy)
        print i
        pass
    print(r)
def tryFindMinAreaRect(paramlist):
    try:
        points((3,))
#         ch_blue,ch_green,ch_red = cv2.split(img_cut)
#         # 直接截去左右20%，留下60%的部分用作识别
#         # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
#         sum_piexl = 0
#         for x in ch_red:
#             for y in x:
#                 sum_piexl += y
#         print(sum_piexl/(32*32))
#         ret,binary_single_ch = cv2.threshold(ch_blue,np.mean(ch_blue),255,cv2.THRESH_BINARY)     
#         cv2.imshow("binary_single_ch",binary_single_ch)
#         # 渐渐减小半径，若可以找到一个半径，这个圆的所有点都是白色，而且圆内部有黑色，则判定该框为装甲板
#         cv2.waitKey()
    except Exception as e:
        logging.exception(e)
        sys._exit(-175)