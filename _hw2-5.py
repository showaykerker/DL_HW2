import time
from collections import namedtuple

import numpy as np
import tensorflow as tf



with open('shakespeare_train.txt', 'r') as f:
    text=f.read()
vocab = sorted(set(text))
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)


batch_size = 64
num_steps = 50
lstm_size = 512
num_layers =2
learning_rate = 0.0001
keep_prob = 0.5
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
        
def build_inputs(batch_size, num_steps):
   
    inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return inputs, targets, keep_prob
    
def build_lstm(lstm_size, num_layers, batch_size, keep_prob):



    def build_cell(lstm_size, keep_prob):

        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop



    cell = tf.contrib.rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)

    return cell, initial_state
    
def build_output(lstm_output, in_size, out_size):


    seq_output = tf.concat(lstm_output, axis=1)
    x = tf.reshape(seq_output, [-1, in_size])


    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))

    logits = tf.matmul(x, softmax_w) + softmax_b
    out = tf.nn.softmax(logits, name='predictions')

    return out, logits
    
    
def build_loss(logits, targets, lstm_size, num_classes):

    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)

    return loss
    
    
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
        



tf.train.get_checkpoint_state('checkpoints')
def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]]=0
    p = p / np.sum(p)
    char = np.random.choice(vocab_size, 1, p=p)[0]
    return char
    
def sample(checkpoint, n_samples, lstm_size, vocab_size, prime='PROTEUS'):
    samples = [c for c in prime]
    model = CharRNN(len(vocab), sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0, 0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction,
                                         model.final_state],
                                         feed_dict=feed)
            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])

            for i in range(n_samples):
                x[0,0] = c
                feed = {model.inputs: x,
                        model.keep_prob: 1.,
                        model.initial_state: new_state}
                preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

                c = pick_top_n(preds, len(vocab))
                samples.append(int_to_vocab[c])

    return ''.join(samples)         


checkpoint = tf.train.latest_checkpoint('checkpoints')
samp = sample(checkpoint, 500, lstm_size, len(vocab), prime='PROTEUS')
print(samp)
