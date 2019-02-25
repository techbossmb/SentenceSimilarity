import word2vec as w2v
import argparse
import numpy as np
import utils
from keras.layers import Input, Dense, LSTM, GRU, RepeatVector, Activation, \
		Dropout, multiply, Lambda, add, Bidirectional, concatenate, Embedding
from keras.models import Model, Sequential, load_model
from keras.callbacks import TensorBoard
import random
import string
import math
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

'''
@author: techbossmb
@date: 06/01/2017
@modified_date: 02/23/2019
'''

class SimilarityModel:
	def __init__(self):
		self.opts = self.load_opts()
		#self.tokenizer = self.build_tokenizer('../data/quora_duplicate_questions.tsv')
	
	def load_opts(self):
		parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		parser.add_argument('--faq_dir', type=str, default='../data/faq', help='dir containing the faqs')
		parser.add_argument('--word2vec_embedding', type=str, default='../data/GoogleNews-vectors-negative300.bin', help='pre-trained word vector embedding')
		parser.add_argument('--embedding_dim', type=int, default=300, help='word2vec dimension for googlenew300 data')
		parser.add_argument('--max_sequence_len', type=int, default=10, help='maximum number of words allowable in query')
		parser.add_argument('--encoding_dim', type=int, default=50, help='LSTM encoding dimension')
		parser.add_argument('--num_epochs', type=int, default=25, help='number of training epochs')
		parser.add_argument('--batch_size', type=int, default=100, help='training batch size')
		parser.add_argument('--dataset', type=str, default='../data/SICK.txt', help='sick dataset filepath')
		opts = parser.parse_args()
		return opts


	def build_tokenizer(self, questions, duplicates, label):
		#datapath = '../data/quora_duplicate_questions.tsv'
		np.random.seed(1337)
		
		all_texts = questions + duplicates
		all_texts = [utils.clean_str(s) for s in all_texts]
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(all_texts)
		# save tokenizer for future use
		with open('tokenizer.tk', 'w') as token_file:
			pickle.dump(tokenizer, token_file)

		sequences = tokenizer.texts_to_sequences(all_texts)
		word_index = tokenizer.word_index
		print('found {} unique tokens'.format(len(word_index)))
		#dataset = pad_sequences(sequences, self.opts.max_sequence_len)
		return tokenizer, word_index

	def build_embedding_layer(self, word2vec_embedding_path, word_index):
		'''build embedding matrix'''

		#word2vec_embedding_path = '../data/glove.6B.300d.txt'
		embeddings_index = {}

		# load pretrained embedding map
		glove_data_path = word2vec_embedding_path
		lines = open(glove_data_path)
		for line in lines:
			values = line.split()
			word = values[0]
			coeffs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coeffs
		lines.close()
		embedding_matrix = np.zeros((len(word_index)+1, self.opts.embedding_dim))
		for word, i in word_index.items():
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector
		# setup embedding layer and freeze it - no retraining of this layer
		embedding_layer = Embedding(len(word_index)+1, self.opts.embedding_dim, weights=[embedding_matrix], input_length=self.opts.max_sequence_len, trainable=False)
		return embedding_layer

	def load_train_val_dataset(self, questions, duplicates, label):
		num_positive_samples = 145000
		num_negative_samples = 145000
		tr_idx = 250000
		val_idx = 290000

		# sample positive and negative labels
		positive_labels = []
		negative_labels = []
		for index, value in enumerate(label):
			if value==1:
				positive_labels.append(index)
			else:
				negative_labels.append(index)
		positive_selection = random.sample(positive_labels, num_positive_samples)
		negative_selection = random.sample(negative_labels, num_negative_samples)
		selection = positive_selection + negative_selection
		random.shuffle(selection)

		#sample questions based on selected labels
		questions = [questions[i] for i in selection]
		duplicates = [duplicates[i] for i in selection]
		label = [label[i] for i in selection] 
		
		questions = [utils.clean_str(sentence) for sentence in questions]
		questions_seq = self.tokenizer.texts_to_sequences(questions)
		questions_data = pad_sequences(questions_seq, self.opts.max_sequence_len)
		
		tr_input_one = questions_data[0:tr_idx]
		val_input_one = questions_data[tr_idx:val_idx]

		duplicates = [utils.clean_str(sentence) for sentence in duplicates]
		duplicates_seq = self.tokenizer.texts_to_sequences(duplicates)
		duplicates_data = pad_sequences(duplicates_seq, self.opts.max_sequence_len)
		tr_input_two = duplicates_data[0:tr_idx]
		val_input_two = duplicates_data[tr_idx:val_idx]

		target = label[0:tr_idx]
		target_val = label[tr_idx:val_idx]
		# convert labels to one-hot encoding
		tr_label = np_utils.to_categorical(target)
		val_label = np_utils.to_categorical(target_val)

		dataset = {
			'tr_input_one': tr_input_one,
			'tr_input_two': tr_input_two,
			'tr_label': tr_label,
			'val_input_one': val_input_one,
			'val_input_two': val_input_two,
			'val_label': val_label
		}
		return dataset

	def build_lstm_model(self, embedding_layer):
		sen_input = Input(shape=(self.opts.max_sequence_len,), dtype='int32')
		dup_input = Input(shape=(self.opts.max_sequence_len,), dtype='int32')	
		embedding_one = embedding_layer(sen_input)
		embedding_two = embedding_layer(dup_input)
		shared_lstm = Bidirectional(LSTM(300), name='bilstm')
		sentence_layer = shared_lstm(embedding_one)
		duplicate_layer = shared_lstm(embedding_two)
		dot_product = multiply([sentence_layer, duplicate_layer])
		minus_duplicate = Lambda(lambda x: -x)(duplicate_layer)
		sentence_duplicate_diff = add([sentence_layer, minus_duplicate])
		diff_squared = multiply([sentence_duplicate_diff, sentence_duplicate_diff])
		features = [sentence_layer, duplicate_layer, dot_product, diff_squared]
		lstm_output = concatenate(features)
		lstm_ouput = Dropout(0.7)(lstm_output)
		sum_of_features = sum(feature.shape[1].value for feature in features)
		layer_size = int(math.floor(math.sqrt(sum_of_features)))*4
		dense_layer = Dense(layer_size, activation='relu')(lstm_output)
		dense_layer = Dropout(0.7)(dense_layer)
		dense_layer = Dense(int(layer_size/2), activation='relu')(dense_layer)
		dense_layer = Dropout(0.6)(dense_layer)
		softmax = Dense(2, activation='softmax', name='softmax')(dense_layer)
		model = Model([sen_input, dup_input], softmax, name='sentence_classifier')
		return model

	def train_model(self, model, dataset, model_params=None):
		tr_input_one = dataset['tr_input_one']
		tr_input_two = dataset['tr_input_two']
		tr_label = dataset['tr_label']
		val_input_one = dataset['val_input_one']
		val_input_two = dataset['val_input_two']
		val_label = dataset['val_label']

		if model_params is None:
			batch_size = 2000
			epochs = 15
			optimizer = 'adam'
		else:
			batch_size = model_params['batch_size']
			epochs = model_params['epochs']
			optimizer = model_params['optimizer']
	
		# generate model identifier
		tensorboard_path = ''.join(random.choice(string.lowercase) for i in range(5))
		print('run tensorboard --logdir=/tmp/{}'.format(tensorboard_path))
		model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
		model.fit([tr_input_one, tr_input_two], tr_label, epochs=epochs, batch_size=batch_size,
			validation_data=([val_input_one, val_input_two], val_label),
			verbose=1,
			callbacks=[TensorBoard(log_dir='/tmp/{}'.format(tensorboard_path))])
		
		model.save('bilstm_model.h5')
		model.save_weights('bilstm_weights.h5')
		return model

	def estimate_model_performance(self, model, dataset):
		val_input_one = dataset['val_input_one']
		val_input_two = dataset['val_input_two']
		val_label = dataset['val_label']

		prediction = model.predict([val_input_one, val_input_two])
		prediction = np.argmax(prediction, axis=1)
		
		tp,fp,tn,fn = 0.0,0.0,0.0,0.0

		for i in range(len(prediction)):
			if prediction[i] == 1:
				if val_label[i] == 1:
					tp += 1
				elif val_label[i] == 0:
					fp += 1
			elif prediction[i] == 0:
				if val_label[i] == 0:
					tn += 1
				elif val_label[i] == 1:
					fn += 1
		print('TP: {}, FP: {}, TN: {}. FN: {}'.format(tp, fp, tn, fn))
		tpr = tp/(tp+fn)
		fpr = fp/(fp+tn)
		tnr = tn/(fp+tn)
		fnr = fn/(tp+fn)
		print('Accuracy: {}'.format((tp+tn)/len(prediction)))
		print('Misclassification: {}'.format((fp+fn)/len(prediction)))
		print('TPR: {}'.format(tpr))
		print('FPR: {}'.format(fpr))
		print('Precision : {}'.format(tp/(fp+tp)))

def main():
	word2vec_embedding_path = '../data/glove.6B.300d.txt'
	quora_dataset_path = '../data/quora_duplicate_questions.tsv'
	# loading quora data approx. 400k datapoints - used for pretraining
	[questions, duplicates, label] = utils.load_quora_data(quora_dataset_path)

	similarity_model = SimilarityModel()
	similarity_model.tokenizer, word_index = similarity_model.build_tokenizer(questions, duplicates, label)
	dataset = similarity_model.load_train_val_dataset(questions, duplicates, label)
	embedding_layer = similarity_model.build_embedding_layer(word2vec_embedding_path, word_index)
	model = similarity_model.build_lstm_model(embedding_layer)
	model = similarity_model.train_model(model, dataset)
	similarity_model.estimate_model_performance(model, dataset)



if __name__=='__main__':
	main()
	print('done')
