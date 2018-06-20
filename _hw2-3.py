import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import time
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

##########################
with open('Dataset/shakespeare_train.txt', 'r') as f:
    text=f.read()
vocab = sorted(set(text))
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
batch_size = 64
num_steps = 50
keep_prob = 0.1
#############################
with open('Dataset/shakespeare_valid.txt', 'r') as f:
    valid=f.read()
valid_encoded = np.array([vocab_to_int[c] for c in valid], dtype=np.int32)    
###################################
def get_batches(arr, n_seqs, n_steps):
    characters_per_batch = n_seqs * n_steps
    n_batches = len(arr)//characters_per_batch
    arr = arr[:n_batches * characters_per_batch]
    arr = arr.reshape((n_seqs, -1))
    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n+n_steps]
        y = np.zeros_like(x)
        y[:, :-1], y[:,-1] = x[:, 1:], x[:, 0]
        yield x, y 
#################################
def build_inputs(batch_size, num_steps):
    inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, keep_prob
####################################
def build_lstm(lstm_size, num_layers, batch_size, keep_prob):

    def build_cell(lstm_size, keep_prob):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop

    cell = tf.contrib.rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)

    return cell, initial_state
####################################    
def build_output(lstm_output, in_size, out_size):

    seq_output = tf.concat(lstm_output, axis=1)
    x = tf.reshape(seq_output, [-1, in_size])

    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
    logits = tf.matmul(x, softmax_w) + softmax_b
    out = tf.nn.softmax(logits, name='predictions')
    return out, logits
#####################################    
def build_loss(logits, targets, lstm_size, num_classes):

    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)

    return loss 
#####################################	
def build_optimizer(loss, learning_rate, grad_clip):

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    return optimizer
    
    
class CharRNN:

    def __init__(self, num_classes, batch_size=64, num_steps=50, lstm_size=128,
                num_layers=2, learning_rate=0.001, grad_clip=5, sampling=False):

        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()

        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)

        x_one_hot = tf.one_hot(self.inputs, num_classes)

        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state

        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)

        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)
epochs = 20

save_every_n = 200

model = CharRNN(len(vocab))

saver = tf.train.Saver(max_to_keep=100)
characters_per_batch = batch_size * num_steps
n_batches = len(encoded)
enc = encoded
enc = enc[:n_batches * characters_per_batch]
display_step = 20
training_iters = epochs * len(enc)
total_display_step = training_iters/(display_step*batch_size)
trainCostAcc= np.arange(total_display_step,dtype = np.float)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    counter = 0
    for e in range(epochs):
        new_state = sess.run(model.initial_state)
        loss = 0
        for x, y in get_batches(encoded, batch_size , num_steps):
            counter += 1
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.loss,
                                                 model.final_state,
                                                 model.optimizer],
                                                 feed_dict=feed)
          
            print('\rEpoch:{}/{}...'.format(e+1, epochs),
                  'Training Step:{}...'.format(counter),
                  'Training loss:{:.4f}...'.format(batch_loss), end='')
           # if (counter % display_step == 0):
           # 	temp = sess.run([model.loss],feed_dict=feed)
           # 	trainCostAcc[counter % display_step] = temp            	
            if (counter % save_every_n == 0):              
                saver.save(sess, 'checkpoints/i{}.ckpt'.format(counter))
    #plt.plot(np.arange(total_display_step),trainCostAcc)
    #plt.show()
    saver.save(sess, 'checkpoints/i{}.ckpt'.format(counter))

