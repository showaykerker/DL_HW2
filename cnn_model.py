import tensorflow as tf
import numpy as np
import time

CONV = {
	'conv1_1': {'filters':16, 'kernel_size':3, 'strides':(1,1), 'activation':None},
	'conv1_2': {'filters':16, 'kernel_size':3, 'strides':(1,1), 'activation':tf.nn.relu},

	'conv2_1': {'filters':32, 'kernel_size':3, 'strides':(1,1), 'activation':None},
	'conv2_2': {'filters':32, 'kernel_size':3, 'strides':(1,1), 'activation':tf.nn.relu},

	'conv3_1': {'filters':64, 'kernel_size':3, 'strides':(1,1), 'activation':None},
	'conv3_2': {'filters':64, 'kernel_size':3, 'strides':(1,1), 'activation':tf.nn.relu},

	'conv4_1': {'filters':128, 'kernel_size':3, 'strides':(1,1), 'activation':None},
	'conv4_2': {'filters':128, 'kernel_size':3, 'strides':(1,1), 'activation':tf.nn.relu},

	'conv5_1': {'filters':128, 'kernel_size':3, 'strides':(1,1), 'activation':None},
	'conv5_2': {'filters':128, 'kernel_size':3, 'strides':(1,1), 'activation':tf.nn.relu},
}

POOL = {
	'pool1': {'pool_size':2, 'strides':2},
	'pool2': {'pool_size':2, 'strides':2},
	'pool3': {'pool_size':2, 'strides':2},
	'pool4': {'pool_size':2, 'strides':2},
	'pool5': {'pool_size':2, 'strides':2},

}

FC = {
	'fc6': 2048,
	'fc7': 2048,
	'fc8': 500,
	'logits': 11,
}

class cnn_model:
	def __init__(self, n_class=11):
		self.n_class=n_class
		self.layers = []


	def build(self, image):
		
		with tf.name_scope('NN'):

			self.layers.append({'0_name':'{:7s}'.format('Input'), '1_shape':'{:16s}'.format(str(image.get_shape()))})

			self.conv1_1 = self.conv_layer(image, "conv1_1")
			self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
			self.pool1   = self.max_pool(self.conv1_2, "pool1")

			tf.summary.histogram('conv1_1', self.conv1_1)
			tf.summary.histogram('conv1_2', self.conv1_2)

			self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
			self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
			self.pool2   = self.max_pool(self.conv2_2, "pool2")

			tf.summary.histogram('conv2_1', self.conv2_1)
			tf.summary.histogram('conv2_2', self.conv2_2)

			self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
			self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
			self.pool3   = self.max_pool(self.conv3_2, "pool3")

			tf.summary.histogram('conv3_1', self.conv3_1)
			tf.summary.histogram('conv3_2', self.conv3_2)

			self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
			self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
			self.pool4   = self.max_pool(self.conv4_2, "pool4")

			tf.summary.histogram('conv4_1', self.conv4_1)
			tf.summary.histogram('conv4_2', self.conv4_2)

			self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
			self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
			self.pool5   = self.max_pool(self.conv5_2, "pool5")

			tf.summary.histogram('conv5_1', self.conv5_1)
			tf.summary.histogram('conv5_2', self.conv5_2)

			self.fc6 = self.dense(self.pool5, "fc6")
			self.fc7 = self.dense(self.fc6  , "fc7")
			self.fc8 = self.dense(self.fc7  , "fc8")

			tf.summary.histogram('fc6', self.fc6)
			tf.summary.histogram('fc7', self.fc7)
			tf.summary.histogram('fc8', self.fc8)
			

			self.logits = self.dense(self.fc8, "logits", n_output=self.n_class, activation=None)

			self.prob = tf.nn.softmax(self.logits)

			return self.prob, self.logits



	def conv_layer(self, input_, name):
		
		with tf.variable_scope(name):
			regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
			conv = tf.layers.conv2d(input_, 
									filters     = CONV[name]['filters'], 
									kernel_size = CONV[name]['kernel_size'],
									strides     = CONV[name]['strides'], 
									padding     = 'SAME', 
									activation  = CONV[name]['activation'],
									kernel_regularizer=regularizer)

			info = {'layer':conv, '0_name':'{:7s}'.format(name), '1_shape':'{:16s}'.format(str(conv.get_shape())), 'n_filters':CONV[name]['filters'], 'strides':CONV[name]['strides'], 'kernel_size':CONV[name]['kernel_size']}
			self.layers.append(info)

			return info['layer']

	def max_pool(self, input_, name):

		ret = tf.layers.max_pooling2d(input_, 
							 pool_size=POOL[name]['pool_size'], 
							 strides=POOL[name]['strides'], 
							 padding='SAME', 
							 name=name)

		info = {'layer':ret, '0_name':'{:7s}'.format(name), '1_shape':'{:16s}'.format(str(ret.get_shape()))}
		self.layers.append(info)

		return info['layer']

	def dense(self, input_, name, activation=tf.nn.relu, n_output=None):

		with tf.variable_scope(name):
			
			shape = input_.get_shape().as_list()
			dim = 1
			for d in shape[1:]: dim *= d
			x = tf.reshape(input_, [-1, dim])

			ret = None

			ret = tf.layers.dense(x, FC[name], activation=activation)
			

			info = {'layer':ret, '0_name':'{:7s}'.format(name), 'n_output':FC[name] , 'activation':str(activation), '1_shape':'{:16s}'.format(str(ret.get_shape()))}
			self.layers.append(info)

			return info['layer']

	def __str__(self):


		ret = '<Class cnn_model>\n' 

		for info in self.layers:
			l = []
			s_k = list(info.keys())

			for k in sorted(s_k):
				l.append((k, info[k]))
			ret = ret + str(l) + '\n'


		return ret

if __name__ == '__main__':
	a = cnn_model(n_class=11)
	img = tf.placeholder(tf.float32, shape=[None, 180, 180, 3])
	a.build(img)
	print(a)
