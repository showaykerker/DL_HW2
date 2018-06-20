from math import pi
import tensorflow as tf
import cv2
import numpy as np

IMAGE_SIZE = 160


def augmentation(data, label, sess):

	#print('\nStart Rotating Images')
	data, label_ = rotate_images(data, label, sess, -30, 30, 2)


	#print('Start Scaling Images')
	#data, label_2 = central_scale_images(data, label_, sess, [1, 0.8, 0.60])


	#print('Start Flipping Images')
	#data, label_2 = flip_images(data, label_, sess)


	#print('Start Adding Noise to Images')
	#data = add_gaussian_noise(data)

	return data, label_


def rotate_images(X_imgs, label, sess, start_angle, end_angle, n_images):
	X_rotate = []
	X_rotate.extend(X_imgs)
	Y_rotate = []
	Y_rotate.extend(label)
	iterate_at = (end_angle - start_angle) / (n_images - 1)
	
	#tf.reset_default_graph()
	X = tf.placeholder(tf.float32, shape = (None, IMAGE_SIZE, IMAGE_SIZE, 3))
	radian = tf.placeholder(tf.float32, shape = (len(X_imgs)))
	tf_img = tf.contrib.image.rotate(X, radian)
	# with tf.Session() as sess:
	# 	sess.run(tf.global_variables_initializer())

	for index in range(n_images):
		degrees_angle = start_angle + index * iterate_at
		radian_value = degrees_angle * pi / 180  # Convert to radian
		radian_arr = [radian_value] * len(X_imgs)
		rotated_imgs = sess.run(tf_img, feed_dict = {X: X_imgs, radian: radian_arr})
		X_rotate.extend(rotated_imgs)
		Y_rotate.extend(label)

	X_rotate = np.array(X_rotate, dtype = np.float32)
	return X_rotate, Y_rotate


def central_scale_images(X_imgs, label, sess, scales):
	# Various settings needed for Tensorflow operation
	boxes = np.zeros((len(scales), 4), dtype = np.float32)
	for index, scale in enumerate(scales):
		x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
		x2 = y2 = 0.5 + 0.5 * scale
		boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
	box_ind = np.zeros((len(scales)), dtype = np.int32)
	crop_size = np.array([IMAGE_SIZE, IMAGE_SIZE], dtype = np.int32)
	
	X_scale_data = []
	Y_scale_data = []
	#tf.reset_default_graph()
	X_scale_holder = tf.placeholder(tf.float32, shape = (None, IMAGE_SIZE, IMAGE_SIZE, 3))
	# Define Tensorflow operation for all scales but only one base image at a time
	tf_img = tf.image.crop_and_resize(X_scale_holder, boxes, box_ind, crop_size)
	# with tf.Session() as sess:
	# 	sess.run(tf.global_variables_initializer())
	
	for img_data in X_imgs:
		batch_img = np.expand_dims(img_data, axis = 0)
		scaled_imgs = sess.run(tf_img, feed_dict = {X_scale_holder: batch_img})
		X_scale_data.extend(scaled_imgs)
		Y_scale_data.extend(label)
	X_scale_data = np.array(X_scale_data, dtype = np.float32)
	return X_scale_data, label


def flip_images(X_imgs, label, sess):
	X_flip = []
	X_flip.extend(X_imgs)
	Y_flip = []
	Y_flip.extend(label)
	#tf.reset_default_graph()
	X_flip_holder = tf.placeholder(tf.float32, shape = ( IMAGE_SIZE, IMAGE_SIZE, 3))
	tf_img1 = tf.image.flip_left_right(X_flip_holder)
	tf_img2 = tf.image.flip_up_down(X_flip_holder)
	tf_img3 = tf.image.transpose_image(X_flip_holder)
	# with tf.Session() as sess:
	# 	sess.run(tf.global_variables_initializer())
	for img in X_imgs:
		flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict = {X_flip_holder: img})
		X_flip.extend(flipped_imgs)
		Y_flip.extend(label)
	X_flip = np.array(X_flip, dtype = np.float32)
	return X_flip, Y_flip
	
def add_gaussian_noise(X_imgs):
	gaussian_noise_imgs = []
	row, col, _ = X_imgs[0].shape
	
	mean = 0
	var = 0.01
	sigma = var ** 0.5
	
	for X_img in X_imgs:
		gaussian = np.random.random((row, col, 1)).astype(np.float32)
		gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
		gaussian_img = cv2.addWeighted(X_img, 0.75, 0.25 * gaussian, 0.25, 0)
		gaussian_noise_imgs.append(gaussian_img)
	gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype = np.float32)
	return gaussian_noise_imgs
