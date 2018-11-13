try:
	import cPickle as pickle
except ImportError:
	import pickle
import os
import sys
import numpy as np
from lasagne import layers
from lasagne.updates import nesterov_momentum, sgd
from nolearn.lasagne import NeuralNet, TrainSplit
from matplotlib import pyplot
from readCSV import loaddata, load2d
from utils import AdjustVariable, plot_sample, draw_loss_2, test
import theano


'''
    Define the structure of CNN.
    Parameters:
        - npochs: number of epochs
'''
def create_network(npochs):
	l_in = layers.InputLayer((None,1,192,256))
	l_conv1 = layers.Conv2DLayer(l_in,num_filters=32,filter_size=(3,3))
	l_pool1 = layers.MaxPool2DLayer(l_conv1,pool_size=(2,2))
	l_drop1 = layers.DropoutLayer(l_pool1,p=0.1)
	l_conv2 = layers.Conv2DLayer(l_drop1,num_filters=64,filter_size=(2,2))
	l_pool2 = layers.MaxPool2DLayer(l_conv2,pool_size=(2,2))
	l_drop2 = layers.DropoutLayer(l_pool2,p=0.2)
	l_conv3 = layers.Conv2DLayer(l_drop2,num_filters=128,filter_size=(2,2))
	l_pool3 = layers.MaxPool2DLayer(l_conv3,pool_size=(2,2))
	l_drop3 = layers.DropoutLayer(l_pool3,p=0.3)
	l_den1 = layers.DenseLayer(l_drop3,num_units=1000)
	l_drop4 = layers.DropoutLayer(l_den1,p=0.5)
	l_den2 = layers.DenseLayer(l_drop4,num_units=1000)
	l_output = layers.DenseLayer(l_den2,num_units=16, nonlinearity=None)
	net = NeuralNet(
		layers=l_output,
			# learning parameters
			update=nesterov_momentum,
			update_learning_rate=theano.shared(np.float32(0.03)),
			update_momentum=theano.shared(np.float32(0.9)),
			regression=True,
			on_epoch_finished=[
				AdjustVariable('update_learning_rate', start=0.03, stop = 0.00001),
				AdjustVariable('update_momentum',start = 0.9, stop = 0.9999),
			],
			max_epochs=npochs, # maximum iteration
			train_split = TrainSplit(eval_size=0.4),
			verbose=1,
		)
	return net
# ==============================================================================

'''
    Train the CNN
    Parameters:
        - ftrain: the path to training data file (csv file)
        - ftest: the path to testing data file (csv file)
        - epochs: number of epochs that we want to train the network
        - savemodel: the name to save the weights and bias after finishing the training (*.pickle file)
        - saveloss, savetest: the file name to save the losses diagram and results on test images.
'''
def train(ftrain,ftest,epochs,savemodel,saveloss,savetest):
	X,y = load2d(ftrain,test=False)
	net3 = create_network(epochs)
	net3.fit(X,y)
	sys.setrecursionlimit(1500000)
	with open(savemodel,'wb') as f:
		pickle.dump(net3,f,-1)
	draw_loss_2(net3,saveloss)
	test(net3,ftest,savetest)

