#!/usr/bin/env python
import rospy
import math
import numpy as np
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
		self.data_dict = pickle.load(open('src/rosneuro_processing_entropy/src/rosneuro_processing_entropy/data/data4(62,5)', 'rb'))

	def configure(self):
		self.buffer = self.configureBuffer()
		self.btfilter = self.configureFilter()
		self.hilb = Hilbert()
		self.nbins = self.data_dict["nbins"]
		self.entropy = ShannonEntropy(self.nbins)
		self.csp_coeff = [None] * self.numBands
		for i in range(self.numBands):
			self.csp_coeff[i] = self.data_dict["csp_coeff_{}".format(i+1)]
		cspdimm = self.data_dict["cspdimm"]
		self.csp = [CommonSpatialPatterns(cspdimm, self.csp_coeff[i]) for i in range(self.numBands)]
		self.mask = self.data_dict["mask"]
		self.clf = self.data_dict["clf"]
		'''self.numClasses = rospy.get_param('/numClasses', default=2)


		self.classLabels = np.empty(self.numClasses)
		for i in range(0, self.numClasses):
			self.classLabels[i] = str(i+1)'''
		
		self.sub_data = rospy.Subscriber(self.sub_topic_data, NeuroFrame, self.onReceivedData)
		self.pub_data = rospy.Publisher(self.pub_topic_data, NeuroOutput, queue_size=1000)

		#self.srv_classify = rospy.ServiceProxy('classify', self.onRequestClassify)
		#self.srv_reset = rospy.ServiceProxy('reset', self.onRequestReset)

		self.new_neuro_frame = False

		#self.msg_.classLabels = self.classLabels

		return True

	def configureBuffer(self):
		self.numChans = self.data_dict["num_chans"]
		self.numSamples = rospy.get_param('/numSamples', default=32)
		self.winLength = self.data_dict["win_length"]
		self.winShift = self.data_dict["win_shift"]
		self.srate = self.data_dict["srate"]
		bufferlen = math.floor(self.winLength*self.srate)
		buffershift = math.floor(self.winShift*self.srate)
		buffer = RingBuffer(bufferlen)

		return buffer

	def configureFilter(self):
		filter_order = self.data_dict["filter_order_list"][0]
		filter_lowf = self.data_dict["filter_lowf"][0]
		filter_highf = self.data_dict["filter_highf"][0]
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
		#labelname = ['STAND', 'WALK']
		t = rospy.Time.now()
		features = []
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
				dcsp = self.csp[i].apply(np.reshape(self.dentropy, (1, len(self.dentropy))))
				features = np.append(features, dcsp[:, self.mask[:,i] == 1])				
			features = np.reshape(features, (1, len(features)))
			self.dproba = self.clf.predict_proba(features)
		else:
			rospy.loginfo('Filling the buffer')
			
		elapsed = (rospy.Time.now() - t).to_sec()
		if elapsed > self.winShift:
			rospy.loginfo('Warning! The loop had a delay of ' + str(elapsed-self.winShift) + ' second')
		else:
			rospy.sleep(self.winShift-elapsed)
		self.new_neuro_frame = False
		return True