#!/usr/bin/env python
from procEntropyTest import SmrBci
import rospy
import math
import numpy as np
import pickle
from rosneuro_msgs.msg import NeuroFrame
from rosneuro_msgs.msg import NeuroOutput
from std_srvs.srv import Empty
from rosneuro_processing_entropy.bciloop_utilities.TimeFilters import ButterFilter
from rosneuro_processing_entropy.bciloop_utilities.Hilbert import Hilbert
from rosneuro_processing_entropy.bciloop_utilities.Entropy import ShannonEntropy
from rosneuro_processing_entropy.bciloop_utilities.SpatialFilters import CommonSpatialPatterns, car_filter
from rosneuro_processing_entropy.bciloop_utilities.RingBuffer import RingBuffer


rospy.init_node("classTest")
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
			print(np.shape(bci.dentropy))
			print(np.shape(bci.final_entropy))

