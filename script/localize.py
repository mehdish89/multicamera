#!/usr/bin/env python

import cv2
import cv2.cv as cv
import numpy as np

import sys
import getopt

from math import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons

import threading
import time

import rospy
from geometry_msgs.msg import PointStamped

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


 
pub = rospy.Publisher('detections', PointStamped, queue_size=10)


"""
def localize(video):
	# img = cv2.imread('b2.jpg')
	# img = cv2.imread('hqdefault.jpg')
	
	# return
	cap = cv2.VideoCapture(video)
"""

# plt.ion()
# 	# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# axes = plt.gca()
# axes.set_xlim([-2,2])
# axes.set_ylim([-1,2])
# axes.set_zlim([-2,2])

bridge = CvBridge()

camera = 0

def handle_frame(msg):
	
	# print(rospy.Time.now())
	# return 
	# ret, img = cap.read()
	global camera

	try:
		img = bridge.imgmsg_to_cv2(msg, "bgr8")
	except CvBridgeError as e:
		print(e)


	img = cv2.medianBlur(img,7)

	h = img.shape[0]
	w = img.shape[1]

	img2 = np.array(img)


	hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

	# lb = np.array([50,110,110])
	# ub = np.array([140,255,255])

	# print(video)


	mn = 110

	lb = np.array([0,mn,60])
	ub = np.array([20,250,250])

	mask1 = cv2.inRange(hsv, lb, ub)

	lb = np.array([160,mn,60])
	ub = np.array([180,250,250])

	mask2 = cv2.inRange(hsv, lb, ub)


	mask = np.maximum(mask1, mask2)

	# print(mask.shape)


	kernel = np.ones((6,6),np.uint8)
	mask = cv2.erode(mask,kernel,iterations = 1)

	kernel = np.ones((5,5),np.uint8)
	mask = cv2.dilate(mask,kernel,iterations = 1)
	

	# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

	# cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

	res = cv2.bitwise_and(img, img, mask= mask)
	edge = cv2.Canny(mask,0,125)

	gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)



	circles = cv2.HoughCircles(edge, cv.CV_HOUGH_GRADIENT, 2, 100, 
		param1=500, param2=30, minRadius=0, maxRadius=0)

	res = (res/2)+(img/2)

	if(circles is None):
		cv2.imshow('detected circles',res)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			exit(0)
		plt.draw()
		return

		# print(circles)

		

	circles = np.uint16(np.around(circles))
	
	for i in circles[0,:]:
	    # draw the outer circle
	    # cv2.circle(res,(i[0],i[1]),i[2],(0,255,0),1)
	    # draw the center of the circle
	    cv2.circle(res,(i[0],i[1]),2,(0,0,255),3)

	    z = i[2]
	    r = 0.018
	    f = 0.018

	    d = f*r*10000/z
	    d = d*1.22/0.27

	    x = i[0]
	    y = i[1]

	    x -= w/2
	    y -= h/2

	    fov = 56

	    rz = fov * x / w
	    ry = fov * y / h


	    px = d * sin(rz*pi/180)
	    py = -d * sin(ry*pi/180)
	    pz = d * cos(ry*pi/180)* cos(rz*pi/180)
	    # print(rz, ry, d)

	    point = PointStamped()
	    point.point.x = px
	    point.point.y = py
	    point.point.z = pz

	    point.header.frame_id = "camera_" + camera

	    # point.header.stamp.secs = rospy.Time.now().secs
	    # point.header.stamp.nsecs = rospy.Time.now().nsecs - 100000

	    pub.publish(point)
	    # print(px, py, pz)

	    # ax.scatter(px, -py, pz, zdir='y')

		# plt.draw()
	
	cv2.imshow('detected circles',res)
	if cv2.waitKey(1) & 0xFF == ord('q'):
			exit(0)
	plt.draw()
	# if cv2.waitKey(1) & 0xFF == ord('q'):
	# 	exit(0)
	# cv2.destroyAllWindows()


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            global camera
            opts, args = getopt.getopt(argv[1:], "h", ["help"])
            print(args)
            camera = str(args[0])
            rospy.init_node('camera_node_' + camera)
            # vn = int(video)
            # localize(str(args[0]))
            
            rospy.Subscriber("image_raw", Image, handle_frame)
            rospy.spin()

            # print(vn)
            # time.sleep(2)


            
        except getopt.error, msg:
             raise Usage(msg)
        # more code, unchanged
    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2

if __name__ == "__main__":
    sys.exit(main())
