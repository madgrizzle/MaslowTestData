from scipy.spatial                           import distance as dist
from imutils                                 import perspective
from imutils                                 import contours
import numpy                                 as np
import glob
import imutils
import cv2
import time
import re
import math


refObj = None
D = 0
currentX, currentY = 0, 0
calX = 0
calY = 0

markerWidth         = 0.25*25.4
counter =0

#def initialize(self):

def midpoint(ptA, ptB):
    return ((ptA[0]+ptB[0])*0.5, (ptA[1]+ptB[1])*0.5)

def removeOutliersAndAverage(data):
    mean = np.mean(data)
    sd = np.std(data)
    tArray = [x for x in data if ( (x >= mean-2.0*sd) and (x<=mean+2.0*sd))]
    return np.average(tArray), np.std(tArray)

def translatePoint(xB, yB, xA, yA, angle):
    if (angle < -45 ) and (angle >-135):
        angle += 90
        angle *= -1.0
    elif (angle > 45 ) and (angle <135):
        angle -= 90
        angle *= -1.0
    elif (angle <= 45 ) and (angle >=-45):
        angle *= -1.0
    elif (angle >= 135 ):
        angle -= 90
    elif (angle <= -135):
        angle += 180
        angle *= -1.0
    cosa = math.cos(angle*3.141592/180.0)
    sina = math.sin(angle*3.141592/180.0)
    xB -= xA
    yB -= yA
    _xB = xB*cosa - yB*sina
    _yB = xB*sina + yB*cosa
    xB = _xB+xA
    yB = _yB+yA
    return xB, yB, math.radians(angle)

def simplifyContour(c):
    tolerance = 0.001
    sides = 20
    while True:
        _c = cv2.approxPolyDP(c, tolerance*cv2.arcLength(c,True), True)
        if len(_c)<=sides or tolerance>=0.5:
            print "First test: len:"+str(len(_c))+", tolerance:"+str(tolerance)
            break
        tolerance += 0.001
    if len(_c)<sides:# went too small.. now lower the tolerance until four points or more are reached
        while True:
            tolerance -= 0.001
            _c = cv2.approxPolyDP(c, tolerance*cv2.arcLength(c,True), True)
            if len(_c)>=sides or tolerance <= 0.01:
                break
    print "len:"+str(len(_c))+", tolerance:"+str(tolerance)
    return _c #_c is the smallest approximation we can find with four our more


files = []

file = "testImages\image2-1.png"

testCount = 0
outFile = open("cameraValues.csv","w")
cv2.namedWindow("image",0)
fileCount = len(glob.glob("circleTestImages\*.png"))
print "filecount:"+str(fileCount)
averageDx = np.zeros([fileCount],dtype=float)
averageDy = np.zeros([fileCount],dtype=float)
averageDi = np.zeros([fileCount],dtype=float)

for file in glob.glob("circleTestImages\*.png"):
#file = "testImages/image2-1.png"
    if (True):
        print file
        image = cv2.imread(file)
        if True:
            height, width, channels = image.shape
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (15, 15), 2, 2)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, width/8, param1=60, param2=40, minRadius=5, maxRadius=100)
            colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0), (255, 0, 255))
            xA = int(width/2)
            yA = int(height/2)

            orig = image.copy()
            print "found "+str(len(circles))+" circles"
            circles = np.around(circles,decimals=3)
            print circles
            minDist = 99999.0
            for i in circles[0,:]:
                cv2.circle(orig,(int(i[0]),int(i[1])),i[2],(0,255,0),2)
                _dist = dist.euclidean((xA,yA), (i[0],i[1]))
                if _dist < minDist:
                    minDist = _dist
                    c = i

            #to determine rotation, find the closest circle to the target circle and calculate the angle
            #determine from this angle if it's above, below, left or right of the target circle
            #and translate the point accordingly
            minDist = 99999.0
            for i in circles[0,:]:
                if (i!=c).any():
                    _dist=dist.euclidean((c[0],c[1]), (i[0],i[1]))
                    if _dist < minDist:
                        minDist = _dist
                        d = i

            cv2.circle(orig,(int(d[0]),int(d[1])),d[2],(0,255,255),2)
            angle = math.atan2(d[1]-c[1], d[0]-c[0])
            print "Computed angle = "+str(math.degrees(angle))
            cv2.imshow("image", orig)

            if c[2] > 10:
                #continue
                cv2.circle(orig,(int(c[0]),int(c[1])),c[2],(255,0,0),2)
                xB = c[0]
                yB = c[1]
                D = c[2]*2.0/markerWidth
                xB,yB,angle = translatePoint(xB,yB,xA,yA,math.degrees(angle))
                print str(xB)+", "+str(yB)+", "+str(math.degrees(angle))
                print "-------"
                print "xA="+str(xA)+", yA="+str(yA)+", xB="+str(xB)+", yB="+str(yB)
                cv2.circle(orig, (int(xA), int(yA)), 10, colors[0], 1)
                cv2.line(orig, (xA, yA-15), (xA, yA+15), colors[0], 1)
                cv2.line(orig, (xA-15, yA), (xA+15, yA), colors[0], 1)
                cv2.circle(orig, (int(xB), int(yB)), 10, colors[3], 1)
                cv2.line(orig, (int(xB), int(yB-15)), (int(xB), int(yB+15)), colors[3], 1)
                cv2.line(orig, (int(xB-15), int(yB)), (int(xB+15), int(yB)), colors[3], 1)
                Dist = dist.euclidean((xA, yA), (xB, yB)) / D
                Dx = dist.euclidean((xA,0), (xB,0))/D
                if (xA>xB):
                    Dx *= -1.0
                Dy = dist.euclidean((0,yA), (0,yB))/D
                if (yA<yB):
                    Dy *= -1.0
                (mX, mY) = midpoint((xA, yA), (xB, yB))
                cv2.putText(orig, file, (15, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[0], 2)
                cv2.putText(orig, "Dx:{:.3f}, Dy:{:.3f}->Di:{:.3f}mm".format(Dx,Dy,Dist), (15, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[0], 2)
                outFile.write("{:.3f}, {:.3f}, {:.3f}\n".format(Dx,Dy,Dist))
                if True:#(Dist>0.25):
                    cv2.imshow("image", orig)
                    cv2.waitKey(0)
                    #cv2.destroyAllWindows()


                    averageDx[testCount] =Dx
                    averageDy[testCount] =Dy
                    averageDi[testCount] =Dist
                    testCount += 1

print "--Dx--"
print averageDx
print "--Dy--"
print averageDy
print "--Di--"
print averageDi
avgDx, stdDx = removeOutliersAndAverage(averageDx)
avgDy, stdDy = removeOutliersAndAverage(averageDy)
avgDi, stdDi = removeOutliersAndAverage(averageDi)
print "AverageDx:"+str(avgDx)+" at "+str(stdDx)+" sd"
print "AverageDy:"+str(avgDy)+" at "+str(stdDy)+" sd"
print "AverageDi:"+str(avgDi)+" at "+str(stdDi)+" sd"
outFile.close()
