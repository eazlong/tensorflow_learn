#!/usr/bin/evn python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

#读取文件，格式为cvs
reader = tf.TextLineReader()
file_name_queue=tf.train.string_input_producer(["testdata.manual.2009.06.14.csv"])
key, value = reader.read(file_name_queue)
print( key, value )
