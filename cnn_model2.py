import tensorflow as tf
import numpy as np
import time

CONV = {
	'conv1' : {'filters': 16, 'kernel_size':3, 'strides':(1,1), 'activation':tf.nn.leaky_relu},
	'conv2' : {'filters': 32, 'kernel_size':3, 'strides':(1,1), 'activation':tf.nn.leaky_relu},
	'conv3' : {'filters': 64, 'kernel_size':3, 'strides':(1,1), 'activation':tf.nn.leaky_relu},
	'conv4' : {'filters':128, 'kernel_size':3, 'strides':(1,1), 'activation':tf.nn.leaky_relu},
	'conv5' : {'filters':256, 'kernel_size':3, 'strides':(1,1), 'activation':tf.nn.leaky_relu},
	'conv6' : {'filters':256, 'kernel_size':3, 'strides':(1,1), 'activation':tf.nn.leaky_relu},
	'conv7' : {'filters':128, 'kernel_size':3, 'strides':(1,1), 'activation':tf.nn.leaky_relu},
	'conv8' : {'filters': 64, 'kernel_size':3, 'strides':(1,1), 'activation':tf.nn.leaky_relu},
	'conv9' : {'filters': 32, 'kernel_size':3, 'strides':(1,1), 'activation':tf.nn.leaky_relu},
	'conv10': {'filters': 16, 'kernel_size':3, 'strides':(1,1), 'activation':tf.nn.leaky_relu},
}

POOL = {
	'pool1': {'pool_size':2, 'strides':2},
	'pool2': {'pool_size':2, 'strides':2},
	'pool3': {'pool_size':2, 'strides':2},
	'pool4': {'pool_size':2, 'strides':2},
	'pool5': {'pool_size':2, 'strides':2},
}

FC = {
	'fc1': {'n_outputs': 256, 'activation': tf.nn.leaky_relu},
	'fc2': {'n_outputs': 128, 'activation': tf.nn.leaky_relu},
	'fc3': {'n_outputs': 32, 'activation': tf.nn.leaky_relu},
	'logits': {'n_outputs': 11, 'activation': None}
}

class cnn_model:
	def __init__(self, n_class=11):
		self.n_class=n_class
		self.layers = []


	def build(self, image):
		
		with tf.name_scope('NN'):

			self.layers.append({'0_name':'{:7s}'.format('Input'), '1_shape':'{:16s}'.format(str(image.get_shape()))})

			self.conv1 = self.conv_layer(image, "conv1")
			self.conv2 = self.conv_layer(self.conv1, "conv2")
			self.pool1 = self.max_pool(self.conv2, "pool1")

			self.conv3 = self.conv_layer(self.pool1, "conv3")
			self.conv4 = self.conv_layer(self.conv3, "conv4")
			self.pool2 = self.max_pool(self.conv4, "pool2")

			self.conv5 = self.conv_layer(self.pool2, "conv5")
			self.conv6 = self.conv_layer(self.conv5, "conv6")
			self.pool3 = self.max_pool(self.conv6, "pool3")
			
			self.conv7 = self.conv_layer(self.pool3, "conv7")
			self.conv8 = self.conv_layer(self.conv7, "conv8")
			self.pool4 = self.max_pool(self.conv8, "pool4")
			
			self.conv9  = self.conv_layer(self.pool4, "conv9")
			self.conv10 = self.conv_layer(self.conv9, "conv10")
			self.pool5  = self.max_pool(self.conv10, "pool5")

			self.fc1   = self.dense(self.pool2, "fc1")
			self.fc2   = self.dense(self.fc1, "fc2")
			self.fc3   = self.dense(self.fc2, "fc3")

			self.logits = self.dense(self.fc3, "logits")

			self.prob = tf.nn.softmax(self.logits)

			tf.summary.histogram("conv1", self.conv1)
			tf.summary.histogram("conv2", self.conv2)
			tf.summary.histogram("conv3", self.conv3)
			tf.summary.histogram("conv4", self.conv4)
			tf.summary.histogram("conv5", self.conv5)
			tf.summary.histogram("conv6", self.conv6)
			tf.summary.histogram("conv7", self.conv7)
			tf.summary.histogram("conv8", self.conv8)
			tf.summary.histogram("conv9", self.conv9)
			tf.summary.histogram("conv10", self.conv10)
			tf.summary.histogram("fc1", self.fc1)
			tf.summary.histogram("fc2", self.fc2)
			tf.summary.histogram("fc3", self.fc3)

			return self.prob, self.logits



	def conv_layer(self, input_, name):
		
		with tf.variable_scope(name):
			regularizer = None #tf.contrib.layers.l2_regularizer(scale=0.1)
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

	def dense(self, input_, name, n_output=None):

		with tf.variable_scope(name):
			
			shape = input_.get_shape().as_list()
			dim = 1
			for d in shape[1:]: dim *= d
			x = tf.reshape(input_, [-1, dim])

			ret = None

			ret = tf.layers.dense(x, FC[name]['n_outputs'], activation=FC[name]['activation'])
			
			info = {'layer':ret, '0_name':'{:7s}'.format(name), 'n_output':FC[name] , 'activation':FC[name]['activation'], '1_shape':'{:16s}'.format(str(ret.get_shape()))}
			self.layers.append(info)

			return info['layer']

	def count(self):
		total_parameters = 0
		for variable in tf.trainable_variables():
			# shape is an array of tf.Dimension
			shape = variable.get_shape()
			#print(shape)
			#print(len(shape))
			variable_parameters = 1
			for dim in shape:
				#print(dim)
				variable_parameters *= dim.value
			#print(variable_parameters)
			total_parameters += variable_parameters
		return total_parameters

	def __str__(self):


		ret = '<Class cnn_model>\n' 

		for info in self.layers:
			l = []
			s_k = list(info.keys())

			for k in sorted(s_k):
				l.append((k, info[k]))
			ret = ret + str(l) + '\n'

		ret = ret + '\n' + 'Total trainable: ' + str(self.count())

		return ret

if __name__ == '__main__':
	a = cnn_model(n_class=11)
	img = tf.placeholder(tf.float32, shape=[None, 160, 160, 3])
	a.build(img)
	print(a)
	