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

class SmrBci:
	def __init__(self):
		self.sub_topic_data = '/neurodata'
		self.pub_topic_data = 'neuroprediction'

	def configure(self):
		self.numChans = rospy.get_param('/numChans', default=16)
		self.numSamples = rospy.get_param('/numSamples', default=32)
		self.winLength = rospy.get_param('/winLength', default=1.5)
		self.winShift = rospy.get_param('/winShift', default=0.125)
		self.srate = rospy.get_param('/srate', default=512)
		bufferlen = math.floor(self.winLength*self.srate)
		self.buffer = RingBuffer(bufferlen)
		filter_order = rospy.get_param('/filter_order', default=4)
		filter_lowf = rospy.get_param('/filter_lowf', default=16)
		filter_highf = rospy.get_param('/filter_highf', default=30)
		self.btfilter = ButterFilter(filter_order, low_f=filter_lowf, high_f=filter_highf, filter_type='bandpass', fs=self.srate)
		self.hilb = Hilbert()
		self.nbins = rospy.get_param('/nbins', default=32)
		self.entropy = ShannonEntropy(self.nbins)
		if rospy.has_param('/csp_coeff'):
			self.csp_coeff = np.load(rospy.get_param('/csp_coeff'))
		else:
			self.csp_coeff = np.load('path-to-csp_coeff')
		cspdimm = rospy.get_param('/cspdimm', default=8)
		self.csp = CommonSpatialPatterns(cspdimm, self.csp_coeff)
		self.clf = pickle.load(open('path-to-clf', 'rb'))
		self.numClasses = rospy.get_param('/numClasses', default=2)


		self.classLabels = np.empty(self.numClasses)
		for i in range(0, self.numClasses):
			self.classLabels[i] = str(i+1)
		
		self.sub_data = rospy.Subscriber(self.sub_topic_data, NeuroFrame, SmrBci.onReceivedData)
		self.pub_data = rospy.Publisher(self.pub_topic_data, NeuroOutput, queue_size=1000)

		self.srv_classify = rospy.ServiceProxy('classify', SmrBci.onRequestClassify)
		self.srv_reset = rospy.ServiceProxy('reset', SmrBci.onRequestReset)

		self.new_neuro_frame = False

		self.msg_.classLabels = self.classLabels

		return True
	
	def getFrameRate(self):
		framerate = 1000.0*self.numSamples/self.srate
		return framerate

	def Reset(self):
		rospy.loginfo('Reset probabilities')
		self.msg_.header.stamp = rospy.Time.now()
		self.pub_data.publish(self.msg_)

	def onReceivedData(self, msg):
		if (msg.eeg.info.nsamples == self.numSamples) and (msg.eeg.info.nchannels == self.numChans):
			self.new_neuro_frame = True
			self.data = msg.eeg.data
			#self.msg.soft_predict.info = self.msg.hard_predict.info = msg.info
	
	def onRequestClassify (self, req, res):
		return SmrBci.Classify()
	
	def onRequestReset(self, req, res):
		SmrBci.Reset()
		return True
	
	def Classify(self):
		if not(self.new_neuro_frame):
			return False
		labelname = ['STAND', 'WALK']
		t = rospy.Time.now()
		chunk = self.data
		self.buffer.append(chunk)
		if self.buffer.isFull:
			data = np.array(self.buffer.data)
			np.clip(data, -400, 400, out=data)
			dcar = car_filter(data, axis=1)
			dfilt = self.btfilter.apply_filt(dcar)
			self.hilb.apply(dfilt)
			denv = self.hilb.get_envelope()
			dentropy = self.entropy.apply(denv)
			dcsp = self.csp.apply(dentropy.reshape(1, len(dentropy)))
			dproba = self.clf.predict_proba(dcsp)
		else:
			rospy.loginfo('Filling the buffer')
		
		elapsed = rospy.Time.now() - t
		if elapsed > self.winShift:
			rospy.loginfo('Warning! The loop had a delay of' + str(elapsed-self.winShift) + ' second')
		else:
			rospy.sleep(self.winShift-elapsed)