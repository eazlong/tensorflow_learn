#!/usr/bin/evn python
# -*- coding:utf-8 -*-

class FNN(object):
    def __init__(self, learning_rate, drop_keep, Layers, N_hidden,
        D_input, D_label, TaskType="regression", L2_lambda=0.0 ):
        self.learning_rate = learning_rate;
        self.drop_keep = drop_keep
		self.Layers = Layers
		self.N_hidden = N_hidden
		self.D_input = D_input
		self.D_label = D_label
		self.Task_type = Task_type
		self.L2_lambda = L2_lambda
		self.L2_penaly = L2_penaly

		with tf.name_scope('Input'):
		  self.inputs = tf.placeholder( tf.float32, [None, D_input], name='input');
		with tf.name_scope('Label'):
		  self.label = tf.placeholder( tf.float32, [None, D_label], name='label')
		with tf.name_scope('KeepRate'):
		  self.drop_keep_rate = tf.placeholder(tf.float32, name='keep_rate')

		self.build('F')

    def weight_init(self, shape):
        initial=tf.truncated_normal(shape, stddev=0.1);
        return tf.Variable(initial)

    def bias_init(self,shape):
          # can change initialization here
          initial = tf.constant(0.1, shape=shape)
          return tf.Variable(initial)

    def variable_summaries(self, var, name):
        with tf.name_scope(name+"_summaries"):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/'+name, mean)
            with tf.name_scope(name+'_stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            # 记录每次训练后变量的数值变化
            tf.scalar_summary('_stddev/' + name, stddev)
            tf.scalar_summary('_max/' + name, tf.reduce_max(var))
            tf.scalar_summary('_min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)

    def layer(self, in_tensor, in_dim, out_dim, layer_name, act=tf.nn.relu ):
    	with tf.name_scope(layer_name):
    		with tf.name_scope(layer_name+"_weight"):
    			weights = weight_init([in_dim, out_dim])
    			self.variable_summaries(weights, layer_name+"_weights")
    		with tf.name_scope(layer_name+'_biases'):
                biases = self.bias_init([out_dim])
                self.variable_summaries(biases, layer_name + '/biases')
            with tf.name_scope(layer_name+'_Wx_plus_b'):
            	pre_activate=tf.matmul(in_tensor, weights)+biases
            	tf.histogram_summary(layer_name + '/pre_activations', pre_activate)
            activations=act(pre_activate, name="activations")
            tf.histogram_summary(layer_name + '/activations', activations)
        return activations,tf.nn.l2_loss(weights)

    def drop_layer(self, intensor):
    	return tf.nn.dropout(in_tensor, self.drop_keep_rate)

    def build(self, prefix):
    	incoming=self.inputs
		if self.Layers!=0:
			layer_nodes = [self.D_input] + self.N_hidden
		else:
			layer_nodes = [self.D_input]
    	# hid_layers用于存储所有隐藏层的输出
		self.hid_layers=[]
		# W用于存储所有层的权重
		self.W=[]
		# b用于存储所有层的偏移
		self.b=[]
		# total_l2用于存储所有层的L2
		self.total_l2=[]
		# drop存储dropout后的输出
		self.drop=[]
		for l in range(self.Layers):
			incoming, l2_loss=self.layer(incoming, layer_nodes[l], layer_nodes[l+1], prefix+'_hid_'+str(l+1),act=tf.nn.relu )
			self.total_l2.append(l2_loss)
			print('Add dense layer: relu with drop_keep:%s' %self.drop_keep)
          	print('    %sD --> %sD' %(layer_nodes[l],layer_nodes[l+1]))
          	self.hid_layers.append(incoming)
          	incoming = self.drop_layer(incoming)
          	self.drop.append(incoming)
      	if self.Task_type=='regression':
          out_act=tf.identity
      	else:
          # 分类任务使用softmax来拟合概率
        	out_act=tf.nn.softmax
      	self.output,l2_loss= self.layer(incoming,layer_nodes[-1],self.D_label, layer_name='output',act=out_act)
      	self.total_l2.append(l2_loss)
      	print('Add output layer: linear')
      	print('    %sD --> %sD' %(layer_nodes[-1],self.D_label))
		if self.Task_type=='regression':
		  with tf.name_scope('SSE'):
		      self.loss=tf.reduce_mean((self.output-self.labels)**2)
		      tf.scalar_summary('loss', self.loss)
		else:
		  # 若为分类，cross entropy的loss function
		  entropy = tf.nn.softmax_cross_entropy_with_logits(self.output, self.labels)
		  with tf.name_scope('cross entropy'):
		      self.loss = tf.reduce_mean(entropy)
		      tf.scalar_summary('loss', self.loss)
		  with tf.name_scope('accuracy'):
		      correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.labels, 1))
		      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		      tf.scalar_summary('accuracy', self.accuracy)
		# 整合所有loss，形成最终loss
		with tf.name_scope('total_loss'):
		  self.total_loss=self.loss + self.l2_penalty*self.L2_lambda
		  tf.scalar_summary('total_loss', self.total_loss)

		# 训练操作
		with tf.name_scope('train'):
		  self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)



inputs=[[0,0],[0,1],[1,0],[1,1]]
outputs=[0,1,1,0]
X=np.array(inputs).reshape((4,1,2)).astype('int16')
Y=np.array(outputs).reshape((4,1,1)).astype('int16')

ff=FNN(learning_rate=1e-3, drop_keep=0.1, Layers=1, N_hidden=2, D_input=2, D_label=1 )
sess=tf.InteractiveSession()
tf.global_variable_initializer().run()
merged=tf.merge_all_summaries()
train_writer=tf.train.SummaryWriter('log' + '/train',sess.graph)

W0=sess.run(ff.W[0])
W1=sess.run(ff.W[1])
# 显示
print('W_0:\n%s' %sess.run(ff.W[0]))
print('W_1:\n%s' %sess.run(ff.W[1]))

pY=sess.run(ff.outputs, feed_dict={ff.inputs:X.reshape((4,2)),ff.drop_keep_rate:1.0})



