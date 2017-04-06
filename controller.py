import os
import data_prepare as dp
import ptb_word_lm as lm
import re
import jieba
import tensorflow as tf

def start_train( p, category, data_file, vocab_file, retrain=False ):
	train_data = None
	word_to_id = None
	vocabs = None
	#如果存在词汇文件，则从文件创建词汇表，否则从训练文件中创建
	if os.path.exists(vocab_file):
		word_to_id, vocabs = dp.build_vocabulary(vocab_file)
	else:
		word_to_id, vocabs = dp.create_vocabulary_from_data_file(vocab_file, data_file)
	train_data = dp.file_to_word_ids(data_file, word_to_id)

	p.train( train_data, vocabs, retrain )

def determine( p, sentenses, vocab_file ):
	word_to_id, vocabs = dp.build_vocabulary(vocab_file)
	sentenses_ids = []
	for s in sentenses:
		data = jieba.lcut(s)
		ids = dp.data_to_word_ids(data, word_to_id)
		sentenses_ids.append(ids)
	return p.determine( sentenses_ids )











