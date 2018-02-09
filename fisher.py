import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def tf_shape(x):
    return x.get_shape().as_list()

def compute_gradients(tensor, var_list):
  grads = tf.gradients(tensor, var_list)
  return [grad if grad is not None else tf.zeros_like(var)
          for var, grad in zip(var_list, grads)]

# def compute_fisher(session, db, logprob, var_list):
#     # fisher_values = []
#     # num_samples = len(db)
#     # for v in range(len(var_list)):
#     #     fisher_values.append(np.zeros(tf_shape(var_list[v])))
#     ders = [ session.run(tf.gradients(logprob, list(var_list)), feed_dict={x: db})  
#     # for v in range(len(var_list)):
#     #     fisher_values[v] += np.square(ders[v]) 
#     # for v in range(len(var_list)):
#     #     fisher_values[v] /= num_samples
#     # 
#     # return fisher_values

# Tensorflow init 
sess = tf.InteractiveSession()

# net init 
n = 40
x = tf.placeholder(tf.float32, [None, n])
w1 = tf.get_variable("w1", initializer=0.2*np.random.rand(n, 20).astype("float32"))
b1 = tf.get_variable("b1", initializer=np.zeros([1, 20], dtype="float32"))
h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
w2 = tf.get_variable("w2", initializer=0.2*np.random.rand(20, 10).astype("float32"))
b2 = tf.get_variable("b2", initializer=np.zeros([1, 10], dtype="float32"))
h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
w3 = tf.get_variable("w3", initializer=0.2*np.random.rand(10, 20).astype("float32"))
b3 = tf.get_variable("b3", initializer=np.zeros([1, 20], dtype="float32"))
h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)
w4 = tf.get_variable("w4", initializer=0.2*np.random.rand(20, n).astype("float32"))
b4 = tf.get_variable("b4", initializer=np.zeros([1, n], dtype="float32"))
y = tf.matmul(h3, w4) + b4

var_list = [w1, w2, w3, w4, b1, b2, b3, b4]
# train 
loss = tf.reduce_mean(tf.pow(x - y, 2))
grads_and_vars = tf.train.GradientDescentOptimizer(0.3).compute_gradients(loss, var_list)
original_grads  = [grad for grad,_ in grads_and_vars]
fisher_grads = tf.gradients(-loss, var_list)
current_grads_and_vars = tf.train.GradientDescentOptimizer(0.1).compute_gradients(loss, var_list=var_list)
current_grads  = [grad for grad,_ in current_grads_and_vars]

fisher_values = tf.placeholder(tf.float32, [v.shape for v in var_list])

ewc_loss = loss + 

# input db
# task: gaussians over n units --- mean ~= n-th unit; sd = 2 units
tasks = []
for k in range(n):
    tasks.append( np.vstack([ np.exp(-((np.arange(n) - k + i)**2)/(2*(5**2)))
        for i in np.random.randn(100)]).astype("float32"))

C = np.vstack(tasks)
C -= C.mean()
C /= C.std()

orig_idcs = np.arange(len(C))
idcs = orig_idcs.copy()


# run
interleaved = False 
epochs = 500
batch_size = 50
sess.run(tf.global_variables_initializer())

fig = plt.figure(figsize=(6,8))
ax = fig.add_subplot(111)
img = ax.imshow(np.zeros(C.shape), vmin=-1, vmax=1, aspect="auto")
for epoch in range(epochs):
    if not interleaved: np.random.shuffle(idcs)    
    CC = C[idcs].copy()
    losses = []
    for batch in range(epochs//batch_size):
        X =  CC[batch*batch_size:(batch+1)*batch_size]
         
        Y, loss_,_,_  = sess.run([y, loss, train, fisher_grads], 
                feed_dict={x: X})
    
        losses.append(loss_)
    loss_mean = np.mean(losses)

    Y  = sess.run(y, feed_dict={x: CC})
    if epoch % 10 == 0:
        print epoch, loss_mean
        idcs_ = CC.argmax(0)
        img.set_data(np.hstack([CC[idcs_], Y[idcs_]])) 
        fig.canvas.draw()
        plt.pause(0.01)
raw_input()
