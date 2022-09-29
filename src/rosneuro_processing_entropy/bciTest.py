#!/usr/bin/env python
from procEntropy import SmrBci
import rospy
import numpy as np


rospy.init_node("bciTest")
bci = SmrBci()
bci.configure()

r=rospy.Rate(256)
while not rospy.is_shutdown():
	if bci.Classify():
		print("Classification started")
		#print(bci.dproba)
	r.sleep()