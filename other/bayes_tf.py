import numpy as np
import data_prepare as dp
import tensorflow as tf
import collections

class classification:
	vocab_list_size = 5000

	__category_set = []
	__data_set = []
	__vocab_list = []

	__sess = tf.Session()
	__y = None
	__x = None

	def __build_dataset_by_category( self, data, category ):
		m = collections.Counter(data).most_common(self.vocab_list_size)
		d = dict(m)
		i = self.__category_set.index(category)
		for val in d.items():
			if  val[0] in self.__data_set[i]:
				self.__data_set[i][val[0]]+=val[1]
				print( val[0] , " exist " )
			else:
				self.__data_set[i][val[0]]=val[1]

	#build a word set with size 'word_count'
	def __build_vocab_list( self, data_in_category ):
		for datas in data_in_category.items():
			self.__category_set.append( datas[0] )
			self.__data_set.append({})
			for data in datas[1]:
				self.__build_dataset_by_category( data, datas[0] )
		vocab_set = set([])
		for data in self.__data_set:
			l = sorted(data.items(), key=lambda data:data[1], reverse=True)
			vocab_set = vocab_set|set([x[0] for x in l])
		self.__vocab_list = list(vocab_set)
		print( self.__vocab_list )

	def __build_word2vec( self, input_data ):
		i = 0
		vec = [0]*len( self.__vocab_list )
		for word in input_data:
			if word in 	self.__vocab_list:
				vec[self.__vocab_list.index(word)] += 1
			else:
				i += 1
		print( 'not in vocab list %d' %(i) )
		return vec

	def train( self, data_in_category ):
		self.__build_vocab_list( data_in_category )

		vec = []
		category_vec = []
		for datas in data_in_category.items():
			for data in datas[1]:
				v = self.__build_word2vec( data )
				vec.append( v )
				cv = [0.0]*len(self.__category_set)
				cv[self.__category_set.index(datas[0])]=1.0;
				category_vec.append(cv)
		print( "vocab_list:", self.__vocab_list )
		print( "vec:", vec )
		#category_probability = tf.Variable( tf.zeros( [category_count] ) )
		#each_word_count_per_category = tf.Variable( tf.ones( [word_count, category_count] ) )
		#total_words_per_category = tf.Variable( tf.ones( [category_count] ) )
		category_count = tf.constant( len(self.__category_set) )
		word_count = tf.constant( len(self.__vocab_list) )
		b = tf.Variable( tf.zeros( [category_count] ) )
		w = tf.Variable( tf.zeros( [word_count, category_count] ) )
		x = tf.placeholder( "float", [None, len(self.__vocab_list)] ) #input document
		y = tf.nn.softmax( tf.matmul( x, w ) + b )
		y_ = tf.placeholder( "float", [None,len(self.__category_set)] )
		cross_entropy = -tf.reduce_sum( y_*tf.log(y) )
		train_step = tf.train.GradientDescentOptimizer( 0.05 ).minimize( cross_entropy )

		init = tf.initialize_all_variables()
		self.__sess.run( init )
		self.__sess.run( train_step, feed_dict={ x:vec[0:3], y_:category_vec[0:3] } )
		print(self.__sess.run( w ))
	
		#print(self.__sess.run( w ))

		self.__x = x
		self.__y = y

	def classify( self, data ):
		v = self.__build_word2vec( data )
		s = self.__sess.run( self.__y, feed_dict={self.__x:[v]} )
		return self.__category_set[s.argmax()]
	

# with tf.Session() as sess:
# 	init = tf.initialize_all_variables()
# 	sess.run( init )
# 	sess.run( train_step, feed_dict={ x:vec[0:3], y_:category_set[0:3] } )
	
