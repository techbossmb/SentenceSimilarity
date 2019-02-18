import word2vec as w2v
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

'''
@author: techbossmb
@date: 06/01/2017
'''


def main():
	# load model options
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
	
	np.random.seed(1337)
	# loading quora data approx. 400k datapoints - used for pretraining
	[questions, duplicates, label] = utils.load_quora_data('../data/quora_duplicate_questions.tsv')
	
	all_texts = questions + duplicates
	all_texts = [utils.clean_str(s) for s in all_texts]
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(all_texts)
  with open('tokenizer.tk', 'w') as token_file:
		pickle.dump(tokenizer, token_file)
	sequences = tokenizer.texts_to_sequences(all_texts)
	word_index = tokenizer.word_index
	print 'found ', len(word_index), 'unique tokens'
	data = pad_sequences(sequences, opts.max_sequence_len)
	
	#build embedding matrix
	embeddings_index = {}
	glove_data_path = '../data/glove.6B.300d.txt'
	f = open(glove_data_path)
	for line in f:
		values = line.split()
		word = values[0]
		coeffs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coeffs
	f.close()
	embedding_matrix = np.zeros((len(word_index)+1, opts.embedding_dim))
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
	# setup embedding layer
	embedding_layer = Embedding(len(word_index)+1, opts.embedding_dim, weights=[embedding_matrix], input_length=opts.max_sequence_len, trainable=False)

	# normalized data to remove class bias
	positive_labels = []
	negative_labels = []
	for index, value in enumerate(label):
		if value==1:
			positive_labels.append(index)
		else:
			negative_labels.append(index)
	positive_selection = random.sample(positive_labels, 145000)
	negative_selection = random.sample(negative_labels, 145000)
	selection = positive_selection + negative_selection
	random.shuffle(selection)
	questions = [questions[i] for i in selection]
	duplicates = [duplicates[i] for i in selection]
	label = [label[i] for i in selection] 
	
	questions = [utils.clean_str(sentence) for sentence in questions]
	questions_seq = tokenizer.texts_to_sequences(questions)
	questions_data = pad_sequences(questions_seq, opts.max_sequence_len)
	tr_idx = 250000
	val_idx = 290000
	tr_input = questions_data[0:tr_idx]
	val_input = questions_data[tr_idx:val_idx]
	

	# load pretrained embedding map
	#embedding = w2v.load_word2vec(opts.word2vec_embedding)
	
	# embed question and duplicate data
	#training_vec = w2v.lookup_embedding(embedding, tr_input, opts, flatten=False)
	#validation_vec = w2v.lookup_embedding(embedding, val_input, opts, flatten=False)
	#training_data = np.array(training_vec)
	#validation_data = np.array(validation_vec)

	duplicates = [utils.clean_str(sentence) for sentence in duplicates]
	duplicates_seq = tokenizer.texts_to_sequences(duplicates)
	duplicates_data = pad_sequences(duplicates_seq, opts.max_sequence_len)
	tr_input_dup = duplicates_data[0:tr_idx]
	val_input_dup = duplicates_data[tr_idx:val_idx]
	##training_vec_dup = w2v.lookup_embedding(embedding, tr_input_dup, opts, flatten=False)
	#validation_vec_dup = w2v.lookup_embedding(embedding, val_input_dup, opts, flatten=False)
	#training_data_dup = np.array(training_vec_dup)
	#validation_data_dup = np.array(validation_vec_dup)
	
	target = label[0:tr_idx]
	target_val = label[tr_idx:val_idx]
	target_ohe = np_utils.to_categorical(target)
	target_val_ohe = np_utils.to_categorical(target_val)
	
	#sen_input = Input(shape=(opts.max_sequence_len, opts.embedding_dim))
	#dup_input = Input(shape=(opts.max_sequence_len, opts.embedding_dim))

	sen_input = Input(shape=(opts.max_sequence_len,), dtype='int32')
	dup_input = Input(shape=(opts.max_sequence_len,), dtype='int32')	

	# generate random path for tensorboard
	logpath = ''.join(random.choice(string.lowercase) for i in range(5))
	print 'run tensorboard --logdir=/tmp/'+logpath
	with open('logpath.txt', 'w') as log:
		log.write(logpath)
	
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
	m = sum(t.shape[1].value for t in features)
	layer_size = int(math.floor(math.sqrt(m)))*4
	print 'dense layer size is '+str(layer_size)
	dense_layer = Dense(layer_size, activation='relu')(lstm_output)
	dense_layer = Dropout(0.7)(dense_layer)
	dense_layer = Dense(int(layer_size/2), activation='relu')(dense_layer)
	dense_layer = Dropout(0.6)(dense_layer)
	softmax = Dense(2, activation='softmax', name='softmax')(dense_layer)
  
	model = Model([sen_input, dup_input], softmax, name='sentence_classifier')
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	model.fit([tr_input, tr_input_dup], target_ohe, epochs=15, batch_size=2000,
         validation_data=([val_input, val_input_dup], target_val_ohe),
         verbose=1,
         callbacks=[TensorBoard(log_dir='/tmp/'+logpath)])
	
	model.save('bilstm_sparse_model_s10.h5')
	model.save_weights('bilstm_sparse_weights_s10.h5')

	prediction = model.predict([val_input, val_input_dup])
	prediction = np.argmax(prediction, axis=1)
	t_count = 0
	f_count = 0
	tp_count = 0
	fn_count = 0
	for i in range(len(prediction)):
		if prediction[i]==1:
			t_count += 1
		elif prediction[i]==0:
			f_count += 1
	for i in range(len(target_val)):
		if target_val[i]==1:
			tp_count += 1
		elif target_val[i]==0:
			fn_count += 1
	print t_count, f_count, tp_count, fn_count
	
	tp = 0.0
	fp = 0.0
	tn = 0.0
	fn = 0.0
	for i in range(len(prediction)):
		if prediction[i] == 1:
			if target_val[i] == 1:
				tp += 1
			elif target_val[i] == 0:
				fp += 1
		elif prediction[i] == 0:
			if target_val[i] == 0:
				tn += 1
			elif target_val[i] == 1:
				fn += 1
	print tp, fp, tn, fn
	tpr = tp/(tp+fn)
	fpr = fp/(fp+tn)
	tnr = tn/(fp+tn)
	fnr = fn/(tp+fn)
	print 'tn:', tn, 'fp:', fp
	print 'fn:', fn, 'tp:', tp
	print 'accuracy:', (tp+tn)/len(prediction)
	print 'misclassification:', (fp+fn)/len(prediction)
	print 'tpr:', tpr
	print 'fpr:', fpr
	print 'precision (yes predicted yes):', tp/(fp+tp) 
	print prediction[100:120], target_val[100:120]

if __name__=='__main__':
	main()
	print 'done'
