import argparse
import numpy as np
import utils
from keras.layers import Input, Dense, LSTM, GRU, RepeatVector, Activation, Dropout, multiply, Lambda, add, Bidirectional, concatenate, Embedding
from keras.models import Model, Sequential, load_model
from keras.callbacks import TensorBoard
import random
import string
import math
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import loadfaqs as faq

'''
@authors:techbossmb
@date: 08/29/17
'''

class SimilarityHelper:

	def __init__(self):
		self.opts = self.programParser()
		self.tokenizer = pickle.load(open(self.opts.token_object, 'rb'))
		self.model = self.loadModel()
		self.faqs_data, self.faqs = self.get_faqs(self.tokenizer, self.opts)
		self.get_faqs(self.tokenizer, self.opts)

		print "Similarity model ready to roll"

	def get_faqs(self, tokenizer, opts):

		'''
		get_faqs loads all faq files in data/faq dir into a single list 
		@argu
			tokenizer: saved tokenizer object
			opts: configuration params
		@returns
			faqs_data: list of faq sentence
			faqs: list of faq (sentence list)
		'''

		faqs = faq.load_faqs(opts.faq_dir)
		clean_faqs = [utils.clean_str(sentence) for sentence in faqs]
		faqs_seq = tokenizer.texts_to_sequences(clean_faqs)
		faqs_data = pad_sequences(faqs_seq, opts.max_sequence_len)

		print 'loaded faqs'
		return faqs_data, faqs

	def get_answers(self, text):

		'''
		get_answers loop through faq files in data/faq dir 
		@returns
			str: response to user's question
		'''

		prediction_result = self.predictSimilarity(text)
		question = max(prediction_result.iteritems(), key=lambda x: x[1])[0]
		answer = faq.load_answers(self.opts.faq_dir, question)
		print "loaded answers"
		return answer

	def user_request(self, tokenizer, text,  opts, faq_size):
		'''
		user_request retrieves user query
		modify as fit - make sure each user question is repeated n times
		(where n= len(faqs)). the idea is to compare the user request 
		to every single faq
		@argu
			tokenizer: saved tokenizer object
			opts: configuration params
			faq_size: len of the loaded faq list
		@returns 
			user_input_list_data: list with repeated user input 
			user_input: question asked by user  to be matched to known faqs
		'''

		user_input = text

	# note: make a list of repeated user input same size as faq
		user_input_list = [user_input for i in range(faq_size)]
		user_input_list = [utils.clean_str(sentence) for sentence in user_input_list]
		user_input_list_seq = tokenizer.texts_to_sequences(user_input_list)
		user_input_list_data = pad_sequences(user_input_list_seq, opts.max_sequence_len)
		return user_input_list_data, user_input

	def programParser(self):

		parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		parser.add_argument('--faq_dir', type=str, default='../data/faq', help='dir containing the faqs')
		parser.add_argument('--max_sequence_len', type=int, default=20, help='maximum number of words allowable in query')
		parser.add_argument('--model_file', type=str, default='../data/bilstm_sparse_model.h5', help='saved sentence similarity model')
		parser.add_argument('--weights_file', type=str, default='../data/bilstm_sparse_weights.h5', help='saved model weight')
		parser.add_argument('--token_object', type=str, default='../data/tokenizer', help='saved tokenizer object file')
		opts = parser.parse_args()

		return opts

	def loadModel(self):

		np.random.seed(1)

		# load saved model
		model = load_model(self.opts.model_file)
		model.load_weights(self.opts.weights_file)
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		print 'trained sentence similarity model loaded from file'

		return model



	def predictSimilarity(self, text):

		np.random.seed(1)
		predicted_output = {}

		# load user input
		[request, user_input] = self.user_request(self.tokenizer, text, self.opts, len(self.faqs_data))

		# predict sentence similarity
		prediction = self.model.predict([self.faqs_data, request])
		predicted_classes = np.argmax(prediction, axis=1)

		print 'question: ', user_input
		print 'see match(es) below'
		match_indexes = [i for i, m in enumerate(predicted_classes) if m==1]

		for i in range(len(match_indexes)):
			predicted_output[self.faqs[match_indexes[i]]] = str(prediction[match_indexes[i]][1])

		return predicted_output

