#     EECS 738
#     HW 3: Says one neuron to another
#     File: NeuralNetworkUtility.py
#     Implementation of a neural network to classify MNIST images

import numpy as np
import random
import matplotlib.pyplot as plt

def plot_images(images):
    "Plot a list of MNIST images."
	
    fig, axes = plt.subplots(nrows=1, ncols=len(images))
    for j, ax in enumerate(axes):
        ax.matshow(images[j][0].reshape(28,28), cmap = plt.cm.binary)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
	
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
	
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))
	
def f(x, W1, W2, B1, B2):
    """Return the output of the network if ``x`` is input image and
    W1, W2, B1 and B2 are the learnable weights. """
    
    Z1 = np.dot(W1,x) + B1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2,A1) + B2
    A2 = sigmoid(Z2)
    return A2

def predict(images, W1, W2, B1, B2):
    predictions = []
    for im in images:
        a = f(im[0], W1, W2, B1, B2)
        predictions.append(np.argmax(a))
    return predictions
	
	
def vectorize_mini_batch(mini_batch):
    """Given a minibatch of (image,lable) tuples of a certain size
    return the tuple X,Y where X contains all of the images and Y contains
    all of the labels stacked horizontally """
    
    mini_batch_x = []
    mini_batch_y = []
    for k in range(0, len(mini_batch)):
        mini_batch_x.append(mini_batch[k][0])
        mini_batch_y.append(mini_batch[k][1])
   
    X = np.hstack(mini_batch_x)
    Y = np.hstack(mini_batch_y)

    return X, Y
	

def SGD(training_data, epochs, mini_batch_size, eta, test_data):
    """Gradient descent. 
    Epochs: the number of times the entire training_data is examined.
    mini_batch_size: the number of images used to approximate the gradient 
    each step of gradient descent.
    eta: the learning rate or the step size.
    test_data: check accuracy of the model against the test_data every epoch.
    """
    n = len(training_data)
    n_test = len(test_data)
    
    # randomize the learnable parameters
    # use np.random.randn(m,n) for appropriate (m,n)
    # use 2-layer neural network with 30-dimensional hidden layer
    W1 = np.random.randn(30, 784)
    W2 = np.random.randn(10, 30)
    B1 = np.random.randn(30, 1)
    B2 = np.random.randn(10, 1)
    
    for j in range(epochs):
        random.shuffle(training_data)
        for k in range(0, n, mini_batch_size):
            # mini_batch of size mini_batch_size
            mini_batch = training_data[k: k+mini_batch_size]
            
            
            # create vectorized input X and labels Y
            X, Y = vectorize_mini_batch(mini_batch)
            
            
            #feed forward(vectorized)
            Z1 = np.dot(W1,X) + B1
            A1 = sigmoid(Z1)
            Z2 = np.dot(W2,A1) + B2
            A2 = sigmoid(Z2)
                    
            # backpropagate(vectorized) 
            # use the four equations of backpropagation
            dZ2 = 1/mini_batch_size*(A2-Y)*sigmoid_prime(Z2)
            dW2 = np.dot(dZ2, A1.T)
            
            # for dB1,dB2 use np.sum with the third argument keepdims=True
            # so that the dimensions do not collapse.
            dB2 = 1/mini_batch_size*np.sum(dZ2, axis = 1, keepdims = True)
            dZ1 = 1/mini_batch_size*np.dot(W2.T, dZ2)* sigmoid_prime(Z1)
            dW1 = np.dot(dZ1, X.T)
            dB1 = 1/mini_batch_size*np.sum(dZ1, axis = 1, keepdims = True)
            
            # update parameters by making a gradient step
            W2 = W2 - eta*dW2
            W1 = W1 - eta*dW1 
            B1 = B1 - eta*dB1
            B2 = B2 - eta*dB2
            
            
        # after every epoch, check the accuracy of the model    
        test_results = [(np.argmax(f(x, W1, W2, B1, B2)), y) for (x, y) in test_data]
        num_correct = sum(int(x == y) for (x, y) in test_results)
        print("Epoch {} : {} / {}".format(j, num_correct, n_test));
    return W1, B1, W2, B2
	
	
def check_accuracy(W1, B1, W2, B2, test_data):

	test_results = [(np.argmax(f(x, W1, W2, B1, B2)), y) for (x,y) in test_data]
	n_test = len(test_data)
	num_correct = sum(int(x == y) for (x,y) in test_results)
	accuracy = (num_correct/n_test)*100
	print("Accuracy is: " + repr(round(accuracy,2)) + "%")
