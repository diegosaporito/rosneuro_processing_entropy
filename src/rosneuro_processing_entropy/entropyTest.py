#!/usr/bin/env python
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

eegdata = []

def onReceivedData(msg):
	if (msg.eeg.info.nsamples == numSamples) and (msg.eeg.info.nchannels == numChans):
		global eegdata
		eegdata.append(msg.eeg.data)

		

rospy.init_node("entropyTest")
sub_topic_data = '/neurodata'

numChans = rospy.get_param('/numChans', default=16)
numSamples = rospy.get_param('/numSamples', default=32)
winLength = rospy.get_param('/winLength', default=1.5)
winShift = rospy.get_param('/winShift', default=0.125)
srate = rospy.get_param('/srate', default=512)
bufferlen = math.floor(winLength*srate)
buffershift = math.floor(winShift*srate)
buffer = RingBuffer(bufferlen)
filter_order = rospy.get_param('/filter_order', default=[4])
filter_lowf = rospy.get_param('/filter_lowf', default=[16])
filter_highf = rospy.get_param('/filter_highf', default=[30])
btfilter = [ButterFilter(filter_order[i], low_f=filter_lowf[i], high_f=filter_highf[i], filter_type='bandpass', fs=srate) for i in range(len(filter_lowf))]
hilb = Hilbert()
nbins = rospy.get_param('/nbins', default=32)
entropy = ShannonEntropy(nbins)
numBands = len(btfilter)
dentropy = []

while not rospy.is_shutdown():
	rospy.loginfo("Waiting for message")
	try:
		rospy.wait_for_message(sub_topic_data, NeuroFrame, 10)
		sub_data = rospy.Subscriber(sub_topic_data, NeuroFrame, onReceivedData)
		rospy.loginfo("Data received")
		rospy.sleep(1)
	except rospy.exceptions.ROSException as e:
		rospy.loginfo(e)
		break

	for i in range(len(eegdata)):
		chunk = np.array(eegdata[i])
		map = np.reshape(chunk, (numSamples, numChans))
		buffer.append(map)
		if buffer.isFull:
			flag = True
			rospy.loginfo("Full buffer")
			data = np.array(buffer.data)
			np.clip(data, -400, 400, out=data)
			dcar = car_filter(data)
			dfilt = btfilter[0].apply_filt(dcar)
			hilb.apply(dfilt)
			denv = hilb.get_envelope()
			dentropy.append(entropy.apply(denv))
			print(np.shape(dentropy))

	#np.save("ent.npy", dentropy)
	#rospy.spin()'''