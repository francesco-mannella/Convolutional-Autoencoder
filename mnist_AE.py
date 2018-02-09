# -*- coding: utf-8 -*-

import numpy as np

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.ion()

# tensorflow
import tensorflow as tf
from convolutional_mlp import MLP, linear

# tensorflow mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
# The number of pixels per side of all images
img_side = 28
# Each input is a raw vector.
# The number of units of the network
# corresponds to the number of input elements
n_mnist_pixels = img_side * img_side
# lengths of datatasets patterns
n_train = mnist.train.num_examples
# convert data into {-1,1} arrays
data = np.vstack(mnist.train.images[:])
labels = mnist.train.labels[:]
data = 2*data - 1

def shuffle_imgs():
    idcs = np.arange(n_train)
    np.random.shuffle(idcs)
    data_ = data[idcs]
    labels_ = labels[idcs]
    return data_, labels_

data, labels = shuffle_imgs()
"""
 Autoencoder

* A multilayered perceptron as the autoencoder network 
    takes samples for the mnist dataset and reproduces them
    
    encoder:  X(28*28) -> H1(512) -> H2(256) -> Z(2)
    decoder:  Z(2)     -> H1(256) -> H2(512) -> Y(28*28)
   
    minimizes:
        D_loss =  mean(sqrt( Y - X)**2))

"""

#-------------------------------------------------------------------------------
# only current needed GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#-------------------------------------------------------------------------------
# set the seed for random numbers generation
import os
current_seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
print "seed:%d" % current_seed
rng = np.random.RandomState(current_seed) 
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#------------------------------------------------------------------------------- 
# Plot
class Plotter(object):  
    def __init__(self, epochs=100):    
        
        self.fig = plt.figure(figsize=(10,8))
        gs = gridspec.GridSpec(8, 10)

        
        self.losses_ax = self.fig.add_subplot(gs[:3,:])
        self.losses_ax.set_title("Reconstruction error")
        self.losses_lines = []
        line, = self.losses_ax.plot(0,0)   
        self.losses_lines.append(line)
        self.labels = ["reconstruction"]  
        self.losses_ax.legend(self.losses_lines, self.labels)  
        self.losses_ax.grid(color='b', linestyle='--', linewidth=0.5)
        self.losses_ax.set_yticks(np.linspace(0.0,0.5, 11))
        self.losses_ax.set_xlim([0, epochs]) 
        self.losses_ax.set_ylim([0.0, 0.5])      
        # data patterns          
        self.data_axes = []
        self.data_imgs = []
        for x in range(5):
            for y in range(5):
                ax = self.fig.add_subplot(gs[3+x, y])
                im = ax.imshow(np.zeros([img_side, img_side]), 
                               vmin=-1, vmax=1)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                self.data_axes.append(ax)
                self.data_imgs.append(im)
 
        # reconstructed patterns          
        self.pattern_axes = []
        self.pattern_imgs = []
        for x in range(5):
            for y in range(5):
                ax = self.fig.add_subplot(gs[3+x, 5 + y])
                im = ax.imshow(np.zeros([img_side, img_side]), 
                               vmin=-1, vmax=1)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                self.pattern_axes.append(ax)
                self.pattern_imgs.append(im)

    def plot(self, R_loss, data_patterns, patterns):
        
        losses = [R_loss]   
        t = len(R_loss)
        self.losses_lines[0].set_data(np.arange(t), R_loss)
                                    
        l = len(patterns)
        
        for x in range(5):
            for y in range(5):
                k = x*5 + y     
                im = self.pattern_imgs[k]
                if k<l:
                    im.set_data(patterns[k].reshape(img_side, img_side))
        
        for x in range(5):
            for y in range(5):
                k = x*5 + y     
                im = self.data_imgs[k]
                if k<l:
                    im.set_data(data_patterns[k].reshape(img_side, img_side))
        self.fig.canvas.draw()
        self.fig.savefig("ae.png")
                 
plotter = Plotter()
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#------------------------------------------------------------------------------- 
#Globals 
epochs = 100
num_samples = 1100
lr = 0.01
train_dropout = 1.0
weight_scale = 0.2
decay = 0.05
#-------------------------------------------------------------------------------
graph = tf.Graph()
with graph.as_default():
    
    drop_out = tf.placeholder(tf.float32, ())
    phase_train = tf.placeholder(tf.bool, ())

    #---------------------------------------------------------------------------
    #  Encoder
    autoencoder_layers     = [n_mnist_pixels] 
    autoencoder_outfuns    = []
    autoencoder_dropout    = []
    autoencoder_copy_from  = []
    autoencoder_batch_norms = []
                              
    autoencoder_layers      += [512,           256,          100           ]
    autoencoder_outfuns     += [tf.nn.relu,    tf.nn.relu,   tf.nn.relu    ]
    autoencoder_copy_from   += [None,          None,         None          ]
    autoencoder_dropout     += [True,          True,         True          ]
    autoencoder_batch_norms += [True,          True,         True          ]
  
    autoencoder_layers      += [512,           256,         n_mnist_pixels ]
    autoencoder_outfuns     += [tf.nn.relu,    tf.nn.relu,  tf.tanh        ] 
    autoencoder_copy_from   += [2,             1,           0              ]
    autoencoder_dropout     += [True,          True,        False          ]
    autoencoder_batch_norms += [True,          True,        True           ]
    
    autoencoder = MLP( scope="autoencoder",
                   lr=lr,
                   bn_decay=decay,
                   weight_scale=weight_scale,
                   outfuns=autoencoder_outfuns, 
                   layers_lens=autoencoder_layers,
                   drop_out=autoencoder_dropout,
                   copy_from=autoencoder_copy_from,
                   batch_norms=autoencoder_batch_norms
                   )   
    data_sample = tf.placeholder(tf.float32, [num_samples, autoencoder_layers[0]])
    decoded_patterns = autoencoder.update(data_sample, drop_out=drop_out, phase_train=phase_train) 
    #---------------------------------------------------------------------------   
    R_loss = tf.sqrt(tf.reduce_mean(tf.pow(data_sample - decoded_patterns, 2)))
    #---------------------------------------------------------------------------
    train =  autoencoder.train(R_loss)
    #---------------------------------------------------------------------------   
    def get_data_sample(t):
        return np.vstack(data[t*num_samples:(t+1)*num_samples])      
    #---------------------------------------------------------------------------   
    #---------------------------------------------------------------------------   
    #---------------------------------------------------------------------------   
    # Tf Session
    with tf.Session(config=config) as session:
            
        # writer = tf.summary.FileWriter("output", session.graph)
        session.run(tf.global_variables_initializer())
        
        R_losses = []
        for epoch in range(epochs):
            
            r_losses = []
            data, labels = shuffle_imgs()
            for t in range(len(data)//num_samples):  
                # reconstruction step -- encoder -> decoder (minimize reconstruction  error)
                curr_data_sample = get_data_sample(t)
                current_decoded_patterns, r_loss, _= session.run(
                    [decoded_patterns, R_loss, train], 
                    feed_dict={data_sample:curr_data_sample, 
                               drop_out: train_dropout, 
                               phase_train: True})  
                r_losses.append(r_loss)
            R_losses.append(np.mean(r_losses))
            
            curr_decoded_patterns = session.run(
                decoded_patterns, 
                feed_dict={data_sample:curr_data_sample, 
                           drop_out: 1.0,
                           phase_train: False})     
                          
            plotter.plot(R_losses, curr_data_sample, curr_decoded_patterns)
                