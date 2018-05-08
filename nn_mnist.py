import gzip
import pickle as cPickle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = cPickle.load(f, encoding='bytes')

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

# print(type(train_x[200]))
# print(train_y)
# print(valid_set[1], ' ', test_set[1])
# ---------------- Visualizing some element of the MNIST dataset --------------

# import matplotlib.cm as cm
import matplotlib.pyplot as plt

"""
plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()  # Let's see a sample
print(train_y[57])
"""
# the neural net!!

train_y = one_hot(train_y, 10)
valid_y = one_hot(valid_y, 10)
test_y = one_hot(test_y, 10)

x = tf.placeholder("float", [None, 784]) # muestras
y_ = tf.placeholder("float", [None, 10]) # etiquetas

W1 = tf.Variable(np.float32(np.random.rand(784, 30)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(30)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(30, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(init)

print ("----------------------")
print ("   Start training...  ")
print ("----------------------")

batch_size = 20

errorTrain = []
errorValid = []

accuracyBefore = 0
accuracyAfter = 0
mejora = 1

epoch = 0;
while (mejora > 0.01 or accuracyAfter < 0.90) and epoch < 30:

    for jj in range(len(train_x) // batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    epoch += 1

    accuracyAfter = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    mejora = np.absolute(accuracyAfter - accuracyBefore)
    accuracyBefore = accuracyAfter

    print("Epoch #:", epoch, "Precisión entrenamiento: ", accuracyAfter)
    print("Epoch #:", epoch, "Precisión validación: ", sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y}))

    aux_train = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys});
    aux_valid = sess.run(loss, feed_dict={x: valid_x, y_: valid_y});

    errorTrain.append(aux_train)
    errorValid.append(aux_valid)

    # print("Epoch #:", epoch, "Error entrenamiento: ", aux_train)
    # print("Epoch #:", epoch, "Error validación: ", aux_valid)

print("Iteraciones: ", epoch)
print("Error de entranamiento: ", errorTrain[epoch-1])
print("Error de validación: ", errorValid[epoch-1])
print("Error de test: ", sess.run(loss, feed_dict={x: test_x, y_: test_y}))
print("Precisión de test: ", sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))

plt.figure()
plt.plot(errorTrain, 'b', linewidth = 2, label = 'Train')
errorValidRectificado = [n/100 for n in errorValid]
plt.plot(errorValidRectificado, 'r', linewidth = 2, label = 'Valid')
plt.xlabel("Iteraciones", fontsize=20)
plt.ylabel("Error", fontsize=20)
plt.legend()
plt.show()

"""
10 iteraciones: prueba con: 5 neuronas capa intermedia: Error train/val: 2.7219372/1778.2432
11 iteraciones: prueba con: 10 neuronas capa intermedia: Error train/val: 2.4007578/1438.1692
16 iteraciones: prueba con: 15 neuronas capa intermedia: Error train/val: 3.0718555/1185.2997
6 iteraciones: prueba con: 20 neuronas capa intermedia: Error train/val: 2.3008187/1402.979
9 iteraciones: prueba con: 25 neuronas capa intermedia: Error train/val: 1.9294244/1261.7598
7 iteraciones: prueba con: 30 neuronas capa intermedia: Error train/val: 1.720573/1187.8867
6 iteraciones: prueba con: 40 neuronas capa intermedia: Error train/val: 2.481775/1223.5477
"""

"""
test_x_muestra = test_x[0:10]
test_y_muestra = test_y[0:10]
result = sess.run(y, feed_dict={x: test_x_muestra})
for b, r in zip(test_y_muestra, result):
    print (b, "-->", r)
print ("----------------------------------------------------------------------------------")
"""