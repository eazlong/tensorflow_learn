import os
import data_prepare as dp
import reader
import ptb_word_lm as lm
import re
import tensorflow as tf
import jieba
import collections
from tensorflow.python.platform import gfile

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

#read data and seg to vocab list		
def read_data( file_name ):
	with open( file_name, encoding='utf-8') as f:
		all_cab = f.read()
		all_cab = re.sub( "[\t\r\n\u3000+\.\!\/_,x$%^*(+\"\']+|[·+——！，。：；》《？、?~@#￥%……&*（）【】”“]+", "", all_cab )
		cabs = jieba.lcut(all_cab)
	return cabs

#write vocabularys to file 
def write_vocabulary( vocab_file, words ):
	with gfile.GFile(vocab_file, mode="wb") as vocab_file:
		for w in words:
			vocab_file.write(bytes(w,'utf-8') + b"\n")

#build vocabulary file from data file
def create_vocabulary_from_data_file( vocab_file, data_file ):
	vocabs = read_data(data_file)
	counter = collections.Counter(vocabs)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	words, _ = list(zip(*count_pairs))
	v = _START_VOCAB+list(words)
	write_vocabulary(vocab_file, v)
	word_to_id = dict(zip(v, range(len(v))))
	return word_to_id, v

#build vocabulary list from file
def build_vocabulary( vocab_file ):
	if gfile.Exists(vocab_file):
		rev_vocab = []
		with gfile.GFile(vocab_file, mode="rb") as f:
			rev_vocab.extend(f.readlines())
		rev_vocab = [line.strip() for line in rev_vocab]
		vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
		return vocab, rev_vocab
	else:
		raise ValueError("Vocabulary file %s not found.", vocab_file)

#data file to word ids
def file_to_word_ids(filename, word_to_id):
	data = read_data(filename)
	return [word_to_id[word] for word in data]

#data to word ids
def data_to_word_ids(data, word_to_id):
	ids = []
	for word in data:
		if word in word_to_id:
			ids.append(word_to_id[word])
		else:
			ids.append(word_to_id[_UNK])
	return ids

#delete the words for classification
# def del_useless_word( data ):
# 	with open( './conf/ignore_words' ) as f:
# 		d = f.read();
# 		useless_words = tf.compat.as_str( d ).split()
# 	for w in data:
# 		if w in useless_words:
# 			data.remove( w )
# 	return data


# with open(file_config) as conf:
# 	for line in conf.readlines():
# 		value=line.strip("\n").split(" ",2)
# 		c=value[0]
# 		file=value[1]
# 		if c in category:
# 			category[c].append(file)
# 		else:
# 			category[c]=[file]

if __name__ == '__main__':
	#create_vocabulary_from_data_file("data/meishi_vocab.txt", "data/meishi.txt")
	build_vocabulary("data/meishi_vocab.txt")
