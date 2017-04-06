#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf 
import numpy as np

example = xxxx
queue = tf.RandomShuffleQueue()
enqueue_op = queue.enqueue(example)
data = queue.dequeue_many(batch_size)
train_op = xxx(data)

qr = tf.QueueRunner( queue, [enqueue_op]*4 )

sess = tf.Session()

cood = tf.Coodinator()
enqueue_thread = qr.create_threads(sess, cood=cood, start=True)

try:
	for step in range(100000):
		if cood.should_stop():
			break;
		sess.run( train_op )
except Exception, e:
	cood.request_stop(e)
finally:
	cood.request_stop()
	cood.join(threads)


