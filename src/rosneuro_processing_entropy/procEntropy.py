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
		self.buffer = self.configureBuffer()
		self.btfilter = self.configureFilter()
		self.hilb = Hilbert()
		self.nbins = rospy.get_param('/nbins', default=32)
		self.entropy = ShannonEntropy(self.nbins)
		if rospy.has_param('/csp_coeff'):
			self.csp_coeff = np.load(rospy.get_param('/csp_coeff'))
		else:
			self.csp_coeff = np.load('src/rosneuro_processing_entropy/src/rosneuro_processing_entropy/csp_coeff_1.npy')
		cspdimm = rospy.get_param('/cspdimm', default=8)
		self.csp = CommonSpatialPatterns(cspdimm, self.csp_coeff)
		self.clf = pickle.load(open('src/rosneuro_processing_entropy/src/rosneuro_processing_entropy/clf', 'rb'))
		self.numClasses = rospy.get_param('/numClasses', default=2)


		self.classLabels = np.empty(self.numClasses)
		for i in range(0, self.numClasses):
			self.classLabels[i] = str(i+1)
		
		self.sub_data = rospy.Subscriber(self.sub_topic_data, NeuroFrame, self.onReceivedData)
		self.pub_data = rospy.Publisher(self.pub_topic_data, NeuroOutput, queue_size=1000)

		#self.srv_classify = rospy.ServiceProxy('classify', self.onRequestClassify)
		#self.srv_reset = rospy.ServiceProxy('reset', self.onRequestReset)

		self.new_neuro_frame = False

		#self.msg_.classLabels = self.classLabels

		return True

	def configureBuffer(self):
		self.numChans = rospy.get_param('/numChans', default=16)
		self.numSamples = rospy.get_param('/numSamples', default=32)
		self.winLength = rospy.get_param('/winLength', default=1.5)
		self.winShift = rospy.get_param('/winShift', default=0.125)
		self.srate = rospy.get_param('/srate', default=512)
		bufferlen = math.floor(self.winLength*self.srate)
		buffershift = math.floor(self.winShift*self.srate)
		buffer = RingBuffer(bufferlen)

		return buffer

	def configureFilter(self):
		filter_order = rospy.get_param('/filter_order', default=[4])
		filter_lowf = rospy.get_param('/filter_lowf', default=[16])
		filter_highf = rospy.get_param('/filter_highf', default=[30])
		btfilter = [ButterFilter(filter_order[i], low_f=filter_lowf[i], high_f=filter_highf[i], filter_type='bandpass', fs=self.srate) for i in range(len(filter_lowf))]
		self.numBands = len(btfilter)

		return btfilter
	
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
		return self.Classify()
	
	def onRequestReset(self, req, res):
		self.Reset()
		return True
	
	def Classify(self):
		if not(self.new_neuro_frame):
			return False
		labelname = ['STAND', 'WALK']
		t = rospy.Time.now()
		chunk = np.array(self.data)
		map = np.reshape(chunk, (self.numSamples, self.numChans))
		self.buffer.append(map)
		if self.buffer.isFull:
			data = np.array(self.buffer.data)
			#np.clip(data, -400, 400, out=data)
			dcar = car_filter(data)
			for i in range(self.numBands):
				dfilt = self.btfilter[i].apply_filt(dcar)
				self.hilb.apply(dfilt)
				denv = self.hilb.get_envelope()
				self.dentropy = self.entropy.apply(denv)
				dcsp = self.csp.apply(np.reshape(self.dentropy, (1, len(self.dentropy))))
				self.dproba = self.clf.predict_proba(dcsp)
		else:
			rospy.loginfo('Filling the buffer')

		'''if self.buffer.isFull:
			temp = np.array(self.dentropy)
			self.final_entropy = np.empty([int(np.shape(self.dentropy)[0]/self.numBands), self.numChans, self.numBands])
			for i in range(self.numBands):
				self.final_entropy[:,:, i] = temp[i::self.numBands,:]'''

		
		elapsed = (rospy.Time.now() - t).to_sec()
		if elapsed > self.winShift:
			rospy.loginfo('Warning! The loop had a delay of' + str(elapsed-self.winShift) + ' second')
		else:
			rospy.sleep(self.winShift-elapsed)
		return True