from gensim.models.word2vec import Word2Vec as w2v
import numpy as np
import utils

'''
@author:techbossmb
'''
	
def load_word2vec(filename):
	global sym
	np.random.seed(1337)
	sym = 2 * (np.random.rand(300)-0.5)
	embedding = w2v.load_word2vec_format(filename, binary=True)
	print 'Loaded word embedding'
	return embedding

def lookup_embedding(w2vmodel, sentences, opts, flatten):
	global embedding
	embedding = w2vmodel
	query_embedding = []
	for sentence in sentences:
		query_embedding.append(embed_sentence(sentence, opts, flatten))
	
	return query_embedding

def embed_sentence(sentence, opts, flatten=False):
	words_vec = []
	maxlen = opts.max_sequence_len
	dimension = opts.embedding_dim
	
	# cut off long sentences
	sentence = sentence[:maxlen] 	

	# pad small sentences with zero
	if len(sentence) < maxlen:
		num_offset = maxlen - len(sentence)
		pad = np.zeros(dimension, dtype=np.float32)
		words_vec = [pad for offset in range(num_offset)]

	for word in sentence:
		try:
			words_vec.append(embedding[word])
		except:
			unk = np.empty(dimension, dtype=np.float32)
			unk[:] = sym #2 * (np.random.rand(dimension)-0.5)
			words_vec.append(unk)
	if flatten:
		words_vec = np.concatenate(words_vec, axis=0)	
	return words_vec		
