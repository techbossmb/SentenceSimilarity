import re
import numpy as np

'''
@authors:techbossmb
@date: 06/01/2017
'''

def clean_str(sentence):
	"""
	replaces special characters and tokens
	Tokenization/sentence cleaning for all datasets
	Based github.com/yoonkim/CNN_sentence/..process_data.py 
	Modified for our specific usage
	"""    
	sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)
	sentence = re.sub(r"\'s", " \'s", sentence)
	sentence = re.sub(r"\'ve", " \'ve", sentence)
	sentence = re.sub(r"n\'t", " n\'t", sentence)
	sentence = re.sub(r"\'re", " \'re", sentence)
	sentence = re.sub(r"\'d", " \'d", sentence)
	sentence = re.sub(r"\'ll", " \'ll", sentence)
	sentence = re.sub(r",", " , ", sentence)
	sentence = re.sub(r"!", " ! ", sentence)
	sentence = re.sub(r"\(", " ", sentence)
	sentence = re.sub(r"\)", " ", sentence)
	sentence = re.sub(r"\?", " ", sentence)
	sentence = re.sub(r"\s{2,}", " ", sentence)
	sentence = re.sub(r"\'", "", sentence)
	sentence = re.sub(r"\`", "", sentence)
	return sentence.strip().lower()


def load_askubuntu_data(source):
	data= []
	with open(source, 'rb') as lines:
		for line in lines:
			data.append(re.split(r'\t+', line)[1])
	return data


def load_quora_data(source):
	question = []
	duplicate = []
	label = []
	with open(source, 'rb') as lines:
		lines.next()
		data = map(lambda x: re.split(r'\t+', x)[3:6], lines)
		question = map(lambda x : x[0], data)
		duplicate = map(lambda x: x[1], data)
		label = map(lambda x: int(x[2]), data)
	print 'Loaded quora dataset'
	return question, duplicate, label

def load_sick_file(source):
	sentence = []
	duplicate = []
	similarity = []
	category = []
	with open(source, 'rb') as lines:
		lines.next()
		data = map(lambda x: re.split(r'\t+', x)[1:12], lines)
		sentence = map(lambda x: x[0], data)
		duplicate = map(lambda x: x[1], data)
		similarity = map(lambda x: float(x[3]), data)
		category = map(lambda x: str(x[10]).rstrip('\r\n'), data)
	return sentence, duplicate, similarity, category

def get_sick_indexes(category):
	tr = []
	val = []
	tst = []
	for i, cat in enumerate(category):
		if cat == 'TRAIN':
			tr.append(i)
		elif cat == 'TRIAL':
			val.append(i)
		elif cat == 'TEST':
			tst.append(i)
		else:
			raise ValueError('error in sick data indexing')
	return tr,val,tst

def get_sick_tr_data(sen, dup, sim, idx):
	[tr_sen, tr_dup, tr_sim] = get_sick_data(sen, dup, sim, idx)
	return tr_sen, tr_dup, tr_sim

def get_sick_val_data(sen, dup, sim, idx):
	[val_sen, val_dup,val_sim] = get_sick_data(sen, dup, sim, idx)
	return val_sen, val_dup,val_sim

def get_sick_tst_data(sen, dup, sim, idx):
	[tst_sen, tst_dup, tst_sim] = get_sick_data(sen, dup, sim, idx)
	return tst_sen, tst_dup, tst_sim

def get_sick_data(sen, dup, sim, idx):
	sentence = [sen[i] for i in idx]
	duplicate = [dup[i] for i in idx]
	similarity = [sim[i] for i in idx]
	return sentence, duplicate, similarity

def euclid_dist(input_vector, embedded_vectors):
	return np.linalg.norm(embedded_vectors - input_vector)


def cosine_dist(input_vector, target_vector):
	return np.dot(input_vector, target_vector) / ( np.linalg.norm(input_vector) * np.linalg.norm(target_vector) )


def compute_similarity(input, target, useCosFunc=True):
	return cosine_dist(input,target) if useCosFunc else euclid_dist(input,target)

def get_features(sen, dup):
	dot_product = np.asarray([np.dot(sen, dup)])
	squared_dist = np.square(sen - dup)
	features = np.concatenate(sen, dup, squared_dist, dot_product)
	return features
	
