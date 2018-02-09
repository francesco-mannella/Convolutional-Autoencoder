# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from convolutional_mlp import MLP, linear
plt.ion()
"""
 Convolutuinal-Deconvolutional Autoencoder

* A multilayered perceptron as the autoencoder network 
    takes samples for the mnist dataset and reproduces them
    
    encoder:  X(28*28) -> H1(28*28*32)  -> H2(14*14*16) -> Z(10)
    decoder:  Z(10)    -> H2(14*14*16)  -> H2(28*28*32) -> Y(28*28)
   
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
tf.set_random_seed(current_seed)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# mnist
# import the mnist class
from mnist import MNIST
# init with the 'data' dir
mndata = MNIST('./data')
# Load data
mndata.load_training()
# The number of pixels per side of all images
img_side = 28
# Each input is a raw vector.
# The number of units of the network
# corresponds to the number of input elements
n_mnist_pixels = img_side * img_side
# lengths of datatasets patterns
n_train = len(mndata.train_images)
# convert data into {-1,1} arrays
data = np.vstack(mndata.train_images[:])/255.0
data = 2*data - 1
np.random.shuffle(data)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#------------------------------------------------------------------------------- 
# Plot
class Plotter(object):  
    def __init__(self, stime):    
        
        self.t = 0
        self.fig = plt.figure(figsize=(10,8))
        gs = gridspec.GridSpec(8, 10)

        
        self.losses_ax = self.fig.add_subplot(gs[:2,:])
        self.losses_ax.set_title("Reconstruction error")
        self.losses_lines = []
        line, = self.losses_ax.plot(0,0)   
        self.losses_lines.append(line)
        self.labels = ["reconstruction"]  
        self.losses_ax.legend(self.losses_lines, self.labels)  
        self.losses_ax.grid(color='b', linestyle='--', linewidth=0.5)
        self.losses_ax.set_yticks(np.linspace(0.0,0.25, 11))
        self.losses_ax.set_xlim([0, stime]) 
        self.losses_ax.set_ylim([0.0, 0.25]) 

        # data data_patterns          
        self.data_pattern_axes = []
        self.data_pattern_imgs = []
        for x in range(5):
            for y in range(5):
                ax = self.fig.add_subplot(gs[3+x, y])
                im = ax.imshow(np.zeros([img_side, img_side]), 
                               vmin=-1, vmax=1)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                self.data_pattern_axes.append(ax)
                self.data_pattern_imgs.append(im)

        # reconstructed patterns          
        self.pattern_axes = []
        self.pattern_imgs = []
        for x in range(5):
            for y in range(5):
                ax = self.fig.add_subplot(gs[3+x,5 + y])
                im = ax.imshow(np.zeros([img_side, img_side]), 
                               vmin=-1, vmax=1)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                self.pattern_axes.append(ax)
                self.pattern_imgs.append(im)

        self.fig2 = plt.figure(figsize=(5, 3))
        self.w1_axes = []
        self.w1_imgs = []
        for x in range(3):
            for y in range(5):
                ax = self.fig2.add_subplot(3, 5, x * 5 + y + 1)
                im = ax.imshow(np.zeros([12, 12]), vmin=-0.1, vmax=0.1)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                self.w1_axes.append(ax)
                self.w1_imgs.append(im)
        plt.subplots_adjust(top=1.0, bottom=0.0, 
                            left=0.0, right=1.0, 
                            hspace=0.0, wspace=0.0)
         
        self.fig3 = plt.figure(figsize=(5,2.5))
        self.w2_axes = []
        self.w2_imgs = []
        for x in range(5):
            for y in range(10):
                ax = self.fig3.add_subplot(5, 10, x * 10 + y + 1)
                im = ax.imshow(np.zeros([20, 20]), vmin=-0.3, vmax=0.3)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                self.w2_axes.append(ax)
                self.w2_imgs.append(im)
        plt.subplots_adjust(top=1.0, bottom=0.0, 
                            left=0.0, right=1.0, 
                            hspace=0.0, wspace=0.0)

        if not os.path.exists("imgs"):
            os.makedirs("imgs")

    def plot(self, R_loss, data, patterns, w0, w1, w2):
       
        losses = [R_loss]   
        t = len(R_loss)
        self.losses_lines[0].set_data(np.arange(t), R_loss)
                                    
        l = len(patterns)
        for x in range(5):
            for y in range(5):
                k = x*5 + y     
                im = self.data_pattern_imgs[k]
                if k<l:
                    im.set_data(data[k].reshape(img_side, img_side))
                    
        for x in range(5):
            for y in range(5):
                k = x*5 + y     
                im = self.pattern_imgs[k]
                if k<l:
                    im.set_data(patterns[k].reshape(img_side, img_side))
        self.fig.canvas.draw()
        self.fig.savefig("imgs/cae.png")
        self.fig.savefig("imgs/cae-{:03d}.png".format(self.t))

        for x in range(3):
            for y in range(5):
                k = x*5 + y     
                im = self.w1_imgs[k]
                data = w1[:,:,x,y].reshape(12, 12)
                im.set_data(data)
        self.fig2.canvas.draw()
        self.fig2.savefig("imgs/cae_w1.png")
        self.fig2.savefig("imgs/cae-w1-{:03d}.png".format(self.t))
        self.t += 1
  
        for x in range(5):
            for y in range(10):
                k = x*10 + y     
                im = self.w2_imgs[k]
                data = w2[:,:,x,y].reshape(20, 20)
                im.set_data(data)
        self.fig3.canvas.draw()
        self.fig3.savefig("imgs/cae_w2.png")
        self.fig3.savefig("imgs/cae-w2-{:03d}.png".format(self.t))
        
        self.t += 1                
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#------------------------------------------------------------------------------- 
#Globals 
epochs = 50
num_samples = 550
lr = 0.8
weight_scale=0.2
decay=0.2
#-------------------------------------------------------------------------------
plotter = Plotter(epochs)
graph = tf.Graph()
with graph.as_default():
    #---------------------------------------------------------------------------
    #  Autoencoder
    phase_train = tf.placeholder(tf.bool, ())
    #input
    autoencoder_layers    = [ [28, 28, 1]  ]

    # encoder
    autoencoder_layers    += [ [14, 14, 3],    [7, 7, 5],         [7, 7, 10],       [10],          ]
    autoencoder_outfuns    = [ tf.nn.relu,     tf.nn.relu,        tf.nn.relu,       tf.nn.relu     ] 
    autoencoder_convs      = [ [3, 3, 1, 3],   [12, 12, 3, 5],    [20, 20, 5, 10],  None           ]
    autoencoder_deconvs    = [ None,           None,              None,             None           ]
    autoencoder_copy_from  = [ None,           None,              None,             None           ]
    autoencoder_strides    = [ 2,              2,                 1,                None           ]
    autoencoder_bn         = [ True,           True,              True,             True           ]

    # decoder
    autoencoder_layers    += [ [7, 7, 10],     [7, 7, 5],         [14, 14, 3],      [28, 28, 1]    ]
    autoencoder_outfuns   += [ tf.nn.relu,     tf.nn.relu,        tf.nn.relu,       tf.tanh        ] 
    autoencoder_convs     += [ None,           None,              None,             None           ]
    autoencoder_deconvs   += [ None,           [20, 20, 5, 10],   [12, 12, 3, 5],   [3, 3, 1, 3]   ]
    autoencoder_copy_from += [ None,           2,                 1,                0              ]
    autoencoder_strides   += [ None,           1,                 2,                2              ]
    autoencoder_bn        += [ True,           True,              True,             True           ]

    autoencoder = MLP(
        scope="Autoencoder",
        lr=lr,
        bn_decay=decay,
        weight_scale=weight_scale,
        convs=autoencoder_convs,
        deconvs=autoencoder_deconvs,
        strides=autoencoder_strides,
        outfuns=autoencoder_outfuns, 
        copy_from=autoencoder_copy_from, 
        layers_lens=autoencoder_layers,
        batch_norms=autoencoder_bn)   
    
    data_sample = tf.placeholder(tf.float32, [num_samples] + autoencoder_layers[0])
    reconstructed_sample = autoencoder.update(data_sample, phase_train=phase_train)      
    #---------------------------------------------------------------------------   
    R_loss = tf.reduce_mean(tf.pow(data_sample - reconstructed_sample, 2.0))
    #---------------------------------------------------------------------------
    R_train =  autoencoder.train(R_loss, optimizer=tf.train.GradientDescentOptimizer)
    #---------------------------------------------------------------------------   
    def get_data_sample(t):
        sample = np.vstack(data[t*num_samples:(t+1)*num_samples])  
        return sample.reshape(num_samples, img_side, img_side, 1)    
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
            np.random.shuffle(data)
            for t in range(len(data)//num_samples):
            
                # reconstruction step -- encoder -> decoder (minimize reconstruction  error)
                curr_data_sample = get_data_sample(t)
                current_decoded_patterns, ll, r_loss, _ = session.run(
                    [reconstructed_sample, autoencoder.layers[-7], R_loss, R_train], 
                    feed_dict={
                        data_sample:curr_data_sample, 
                        phase_train: True})       
                r_losses.append(r_loss) 
                                
            R_losses.append(np.mean(r_losses))
            
            np.random.shuffle(data)
            test_data_sample = get_data_sample(0)
            test_decoded_patterns, w0, w1, w2 = session.run( 
                [reconstructed_sample, autoencoder.weights[0], autoencoder.weights[1],
                  autoencoder.weights[2]],
                feed_dict={
                    data_sample:test_data_sample, 
                    phase_train: False}) 
            
            plotter.plot(R_losses, test_data_sample, test_decoded_patterns, w0, w1, w2)
                
