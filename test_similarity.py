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

def get_faqs(tokenizer, opts):
	'''
	get_faqs loads all faq files in data/faq dir into a single list 
	@argu
		tokenizer: saved tokenizer object
		opts: configuration params
	@return list of faq sentence
	'''
	faqs = faq.load_faqs(opts.faq_dir)
	faqs = [utils.clean_str(sentence) for sentence in faqs]
	faqs_seq = tokenizer.texts_to_sequences(faqs)
	faqs_data = pad_sequences(faqs_seq, opts.max_sequence_len)
	print 'loaded faqs'
	return faqs_data

def user_request(tokenizer, opts, faq_size)
	'''
	user_request retrieves user query
	@argu
		tokenizer: saved tokenizer object
		opts: configuration params
		faq_size: len of the loaded faq list
	@return list with repeated user input 
	'''
	user_input = raw_input('Enter your question: ')
	# note: make a list of repeated user input same size as faq
	user_input_list = [user_input for i in range(faq_size)]
	user_input_list = [utils.clean_str(sentence) for sentence in user_input_list]
	user_input_list_seq = tokenizer.texts_to_sequences(user_input_list)
	user_input_list_data = pad_sequences(user_input_list_seq, opts.max_sequence_len)
	return user_input_list_data
	
def main():
	# load program options
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--faq_dir', type=str, default='../data/faq', help='dir containing the faqs')
	parser.add_argument('--max_sequence_len', type=int, default=20, help='maximum number of words allowable in query')
	parser.add_argument('--model_file', type=str, default='../data/bilstm_sparse_model.h5', help='saved sentence similarity model')
	parser.add_argument('--weights_file', type=str, default='../data/bilstm_sparse_weights.h5', help='saved model weight')
	parser.add_argument('--token_object', type=str, default='../data/tokenizer', help='saved tokenizer object file')
	opts = parser.parse_args()
	
	np.random.seed(1)
	
	tokenizer = pickle.load(open(opts.token_object, 'rb'))
	
	# load default faqs
	faqs_data = get_faqs(tokenizer, opts)
	
	# load user input
	request = user_request(tokenizer, opts, len(faqs_data))
	
	# load saved model
	model = load_model(opts.model_file)
	model.load_weights(opts.weights_file)
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	print 'trained sentence similarity model loaded from file'

	# predict sentence similarity
	prediction = model.predict([faqs_data, request])
	predicted_classes = np.argmax(prediction, axis=1)

	match_indexes = [i for i, m in enumerate(predicted_classes) if m==1]
	for i in range(len(match_indexes)):
		print questions[match_indexes[i]], ' -> p('+prediction[match_indexes[i]][1]+')'

if __name__=='__main__':
	main()
	print 'done'
