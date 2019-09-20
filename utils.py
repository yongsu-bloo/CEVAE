import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import sys
from tensorflow.contrib.layers.python.layers import initializers

def fc_net(inp, layers, out_layers, scope, lamba=1e-3, activation=tf.nn.relu, reuse=None,
           weights_initializer=initializers.xavier_initializer(uniform=False),
           data_type="cont"):
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=activation,
                        normalizer_fn=None,
                        weights_initializer=weights_initializer,
                        reuse=reuse,
                        weights_regularizer=slim.l2_regularizer(lamba)):

        if layers:
            h = slim.stack(inp, slim.fully_connected, layers, scope=scope)
            if not out_layers:
                return h
        else:
            h = inp
        outputs = []
        for i, (outdim, activation) in enumerate(out_layers):
            if data_type == 'bin' and i == len(out_layers) - 1:
                o1 = slim.fully_connected(h, outdim, activation_fn=tf.nn.sigmoid, scope=scope + '_{}'.format(i + 1))
            else:
                o1 = slim.fully_connected(h, outdim, activation_fn=activation, scope=scope + '_{}'.format(i + 1))
            outputs.append(o1)
        return outputs if len(outputs) > 1 else outputs[0]


def get_y0_y1(sess, y, f0, f1, shape=(), L=1, verbose=True, task='ihdp'):
    y0, y1 = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    if False:
        y0 += sess.run(y, feed_dict=f0)
        y1 += sess.run(y, feed_dict=f1)
        y0 = np.floor(y0)
        y1 = np.floor(y1)
    else:
        ymean = y.mean()
        for l in range(L):
            if L > 1 and verbose:
                sys.stdout.write('\r Sample {}/{}'.format(l + 1, L))
                sys.stdout.flush()
            y0 += sess.run(ymean, feed_dict=f0) / L
            y1 += sess.run(ymean, feed_dict=f1) / L
    if L > 1 and verbose:
        print()
    return y0, y1
