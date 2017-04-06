import numpy as np
import data_prepare as dp
import tensorflow as tf
import collections

class classification:
	vocab_list_size = 20000

	_category_set = []
	_data_set = []
	_vocab_list = []

	_sess = tf.Session()
	_y = None
	_x = None

	def _build_dataset_by_category( self, data, category ):
		m = collections.Counter(data).most_common(self.vocab_list_size)
		d = dict(m)
		i = self._category_set.index(category)
		for val in d.items():
			if  val[0] in self._data_set[i]:
				self._data_set[i][val[0]]+=val[1]
				print( val[0] , " exist " )
			else:
				self._data_set[i][val[0]]=val[1]

	#build a word set with size 'word_count'
	def _build_vocab_list( self, data_in_category ):
		for datas in data_in_category.items():
			self._category_set.append( datas[0] )
			self._data_set.append({})
			for data in datas[1]:
				self._build_dataset_by_category( data, datas[0] )
		vocab_set = set([])
		for data in self._data_set:
			l = sorted(data.items(), key=lambda data:data[1], reverse=True)
			vocab_set = vocab_set|set([x[0] for x in l])
		self._vocab_list = list(vocab_set)
		print( self._vocab_list )

	def _build_word2vec( self, input_data ):
		i = 0
		vec = [0]*len( self._vocab_list )
		for word in input_data:
			if word in 	self._vocab_list:
				vec[self._vocab_list.index(word)] += 1
			else:
				i += 1
		print( 'not in vocab list %d' %(i) )
		return vec

	def train( self, data_in_category ):
		self._build_vocab_list( data_in_category )

		vec = []
		category_vec = []
		for datas in data_in_category.items():
			for data in datas[1]:
				v = self._build_word2vec( data )
				vec.append( v )
				cv = [0.0]*len(self._category_set)
				cv[self._category_set.index(datas[0])]=1.0;
				category_vec.append(cv)
		print( "vocab_list:", self._vocab_list )
		print( "vec:", vec )
		#category_probability = tf.Variable( tf.zeros( [category_count] ) )
		#each_word_count_per_category = tf.Variable( tf.ones( [word_count, category_count] ) )
		#total_words_per_category = tf.Variable( tf.ones( [category_count] ) )
		category_count = tf.constant( len(self._category_set) )
		word_count = tf.constant( len(self._vocab_list) )
		x = tf.placeholder( "float", [None, len(self._vocab_list)] ) #input document
		b = tf.Variable( tf.zeros( [category_count] ) )
		w = tf.Variable( tf.zeros( [word_count, category_count] ) )
		y = tf.nn.relu( tf.matmul( x, w ) + b )
		y_ = tf.placeholder( "float", [None,len(self._category_set)] )
		cross_entropy = -tf.reduce_sum( y_*tf.log(y) )
		train_step = tf.train.GradientDescentOptimizer( 0.05 ).minimize( cross_entropy )

		init = tf.initialize_all_variables()
		self._sess.run( init )
		self._sess.run( train_step, feed_dict={ x:vec[0:3], y_:category_vec[0:3] } )
		print(self._sess.run( w ))
	
		#print(self._sess.run( w ))

		self._x = x
		self._y = y

	def classify( self, data ):
		v = self._build_word2vec( data )
		s = self._sess.run( self._y, feed_dict={self._x:[v]} )
		return self._category_set[s.argmax()]
	

# with tf.Session() as sess:
# 	init = tf.initialize_all_variables()
# 	sess.run( init )
# 	sess.run( train_step, feed_dict={ x:vec[0:3], y_:category_set[0:3] } )
	
def neural_network():
	with tf.device('/cpu:0'),tf.name_scope("embedding"):
		embedding_size=128
		W=tf.Variable(tf.random_uniform([input_size, embedding_size], -1.0, 1.0))
		embedding_chars=tf.embedding_lookup(W,X)
		embedded_chars_expanded=tf.expand_dims(embedding_chars, -1)
	num_filter=128
	filter_size=[3,4,5]
	pooled_output=[]
	for i, filter_size in enumerate(filter_sizes):
		with tf.name_scope("conv_maxpool_%s" %filter_size ):
			filter_shape=[filter_size,embedding_size,1,num_filter]
			W=tf.Variable(tf.truncated_normal(filter_shape, stdev=0.1))
			b=tf.Variable(tf.constant(0.1,shape=[num_filter]))
			conv=tf.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID")
			h=tf.nn.relu(tf.nn.bias_add(conv, b))
			pooled=tf.nn.max_pool(h, ksize=[1, input_size-file_size,1,1],strides=[1, 1, 1, 1], padding="VALID")
			pooled_output.append(pooled)

	num_filters_total = num_filters * len(filter_sizes)
	h_pool = tf.concat(3, pooled_outputs)
	h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
	# dropout
	with tf.name_scope("dropout"):
		h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
	# output
	with tf.name_scope("output"):
		W = tf.get_variable("W", shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
		b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
		output = tf.nn.xw_plus_b(h_drop, W, b)
		
	return output