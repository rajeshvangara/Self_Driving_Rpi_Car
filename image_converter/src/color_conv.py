#!/usr/bin/env python
from __future__ import print_function

import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class image_converter:

  def __init__(self):
    #self.image_pub = rospy.Publisher("image_topic_2",Image)
	self.bridge = CvBridge()
	self.image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,self.callback)
	def empty(a):
		pass
		
	cv2.namedWindow("HSV")
	cv2.resizeWindow("HSV", 640, 240)
	cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)
	cv2.createTrackbar("HUE Max", "HSV", 179, 179, empty)
	cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)
	cv2.createTrackbar("SAT Max", "HSV", 255, 255, empty)
	cv2.createTrackbar("VALUE Min", "HSV", 0, 255, empty)
	cv2.createTrackbar("VALUE Max", "HSV", 255, 255, empty)

  def callback(self,data):
    try:
      img = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    (rows,cols,channels) = img.shape
    if cols > 60 and rows > 60 :
		cv2.circle(img, (50,50), 10, 255)
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")

    print(h_min)
 
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
 
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    hStack = np.hstack([img, mask, result])
    cv2.imshow('Horizontal Stacking', hStack)
#    if cv2.waitKey(1) and 0xFF == ord('q'):
#        break
	
    #cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)
    

    

def main(args):
  ic = image_converter()
  #print( "WORK")
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
