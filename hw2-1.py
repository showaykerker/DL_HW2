import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from cnn_model2 import cnn_model
import sklearn.preprocessing
from Dataset.augmentation import augmentation
import time
import os
import cv2

time_ = time.strftime("%m%d%H%M", time.localtime())
load_path = 'Dataset/food-11_180.pkl'
log_path = './log/'+time_+'__with_regularization/'

if not os.path.exists(log_path): os.makedirs(log_path)


tf.logging.set_verbosity(tf.logging.INFO)


config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.7


LR = 0.001
n_STEPS = 8000
BATCH_SIZE = 32
DISPLAY_EVERY = 100

with open(load_path, 'rb') as f:
	data = pickle.load(f)

label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(11))

X_train, Y_train = data['X_train'], data['Y_train'] #label_binarizer.transform(data['Y_train'])
X_test, Y_test = data['X_test'], data['Y_test'] #label_binarizer.transform(data['Y_test'])


# for i in range(0, 1000, 100):
# 	cv2.imshow('x', X_train[i])
# 	cv2.waitKey(0)

model = cnn_model()

X = tf.placeholder(tf.float32, [None, 160, 160, 3])
Y = tf.placeholder(tf.int64, [None, ])
prediction, logits = model.build(X)

#tf.summary.image('X', X, max_outputs=3)

saver = tf.train.Saver()

# loss_op = - tf.reduce_sum(Y*tf.log(prediction+1e-10))
# train_op = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss_op)


loss_op = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=Y, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=LR)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.image('input', X)

tf.summary.scalar('loss', loss_op)
tf.summary.scalar('acc', accuracy)


merge_op = tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
	writer = tf.summary.FileWriter(log_path+'ckp_test', sess.graph, flush_secs=10)
	train_writer = tf.summary.FileWriter(log_path+'ckp_train', flush_secs=10)
	sess.run(init)
	print('\n\n\n')
	print('Total trainable:', model.count())
	for step in range(1, n_STEPS+1):
		idx = np.random.choice(len(X_train), BATCH_SIZE)
		batch_X, batch_Y = [X_train[i] for i in idx ], [Y_train[i] for i in idx ]
		batch_X, batch_Y = augmentation(batch_X, batch_Y, sess)
		# for i in range(0, len(batch_X), 100):
		# 	cv2.imshow('', batch_X[i])
		# 	cv2.waitKey(1)
		sess.run(train_op, feed_dict={X:batch_X, Y:batch_Y})
		if step % DISPLAY_EVERY == 0 or step == 1:
			idx_test = np.random.choice(len(X_test), int(BATCH_SIZE))
			batch_X_test, batch_Y_test = [X_test[i] for i in idx_test], [Y_test[i] for i in idx_test]
			summary_train, loss, acc = sess.run([merge_op, loss_op, accuracy], feed_dict={X:batch_X, Y:batch_Y})
			summary, loss_t, acc_t = sess.run([merge_op, loss_op, accuracy], feed_dict={X:batch_X_test, Y:batch_Y_test})
			train_writer.add_summary(summary_train, step)
			writer.add_summary(summary, step)

			print('Step %d:  loss = %6.4f, acc = %6.4f ; test_loss = %6.4f, test_acc = %6.4f'% (step, loss, acc, loss_t, acc_t) )
			saver.save(sess, log_path + "model_%d.ckpt" % step)
