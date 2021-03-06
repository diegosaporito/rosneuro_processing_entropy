#!/usr/bin/env python
from procEntropy import SmrBci
import rospy
import numpy as np
from rosneuro_msgs.msg import NeuroFrame
from rosneuro_msgs.msg import NeuroOutput
from rosneuro_processing_entropy.bciloop_utilities.TimeFilters import ButterFilter
from rosneuro_processing_entropy.bciloop_utilities.Hilbert import Hilbert
from rosneuro_processing_entropy.bciloop_utilities.Entropy import ShannonEntropy
from rosneuro_processing_entropy.bciloop_utilities.SpatialFilters import CommonSpatialPatterns, car_filter
from rosneuro_processing_entropy.bciloop_utilities.RingBuffer import RingBuffer


rospy.init_node("bciTest")
bci = SmrBci()
bci.configure()

while not rospy.is_shutdown():
	try:
		rospy.wait_for_message(bci.sub_topic_data, NeuroFrame, 5)
		rospy.loginfo("Data received")
		rospy.sleep(1)
	except rospy.exceptions.ROSException as e:
		rospy.loginfo(e)
		break
	if bci.Classify():
		if bci.buffer.isFull:
			#print(bci.dentropy)
			print(bci.dproba)