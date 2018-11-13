try:
	import cPickle as pickle
except ImportError:
	import pickle
import os
import lasagne
import sys
import numpy as np
from lasagne import layers
from nolearn.lasagne import NeuralNet, TrainSplit
from matplotlib import pyplot
from readCSV import loaddata, load2d
from utils import AdjustVariable, plot_sample, draw_loss, write_file
from lasagne.layers import DenseLayer
import theano

# CONSTANT: the path to the trained model
FMODEL = '/home/linh/Examples/trained_models/trained_Beetles/cnnmodel3_all_10000_epochs_.pickle'

'''
    Build the layers that have the same ordered with the trained model
'''

def build_model():
	net = {}
	net['input'] = lasagne.layers.InputLayer((None,1,192,256))
	net['conv1'] = lasagne.layers.Conv2DLayer(net['input'], 32, (3,3))
	net['pool1'] = lasagne.layers.MaxPool2DLayer(net['conv1'],pool_size=(2,2))
	net['drop2'] = lasagne.layers.DropoutLayer(net['pool1'],p=0.1)
	net['conv2'] = lasagne.layers.Conv2DLayer(net['drop2'], 64, (2,2))
	net['pool2'] = lasagne.layers.MaxPool2DLayer(net['conv2'],pool_size=(2,2))
	net['drop3'] = lasagne.layers.DropoutLayer(net['pool2'],p=0.2)
	net['conv3'] = lasagne.layers.Conv2DLayer(net['drop3'], 128, (2,2))
	net['pool3'] = lasagne.layers.MaxPool2DLayer(net['conv3'],pool_size=(2,2))
	net['drop4'] = lasagne.layers.DropoutLayer(net['pool3'],p=0.3)
	net['hidden4'] = lasagne.layers.DenseLayer(net['drop4'],num_units=1000)
	net['drop5'] = lasagne.layers.DropoutLayer(net['hidden4'],p=0.5)
	net['hidden5'] = lasagne.layers.DenseLayer(net['drop5'],num_units=1000)
	net['output'] = lasagne.layers.DenseLayer(net['hidden5'],num_units=16,nonlinearity=None)
	return net

'''
    Load the trained model and copy the parameter values into the corresponding layers ( function build_model).
    Then, change the output of the last layer to fine-tune the trained model

    Parameters:
        - model_file: trained model file
'''
def set_weights(model_file):
	with open(model_file) as f:
		model = pickle.load(f)
	print('Set the weights...')
	#newnet = model
	print(model)
	all_param = lasagne.layers.get_all_param_values(model.layers)
	net = build_model()
	lasagne.layers.set_all_param_values(net['output'],all_param,trainable=True)
	output_layer = lasagne.layers.DenseLayer(net['hidden5'],num_units = 20, nonlinearity=None)
	
	return output_layer	

'''
    Build the fine_tuning model after load the trained model and change the output
    Parameters:
        - nlayers: list of layers after copy the values from trained model
'''
def build_fine_tuning_model(nlayers):
	net3 = NeuralNet(
	layers=nlayers,

		# learning parameters
		update= lasagne.updates.nesterov_momentum,
		#update_learning_rate=theano.shared(np.float32(0.1)),
		#update_momentum=theano.shared(np.float32(0.9)),
		update_learning_rate = 0.01,
		update_momentum = 0.9,
		regression=True,
		#on_epoch_finished = [
		#	AdjustVariable('update_learning_rate', start = 0.1, stop = 0.0001),
		#	AdjustVariable('update_momentum', start = 0.9, stop = 0.9999),
		#],
		max_epochs=10000, # maximum iteration
		train_split = TrainSplit(eval_size=0.4),
		verbose=1,
	)
	return net3

'''
def build_model2(modelfile):
	with open(modelfile) as f:
		model = pickle.load(f)
	print('Set the weights...')
	print(model)
	all_param = lasagne.layers.get_all_param_values(model.layers)
	net = build_model()
	lasagne.layers.set_all_param_values(net['output'],all_param,trainable=True)
	newlayers = lasagne.layers.DenseLayer(net['hidden5'],num_units = 16, nonlinearity=None)
	#model.layers = newlayers
	print(model)
	return model
'''

if __name__ == '__main__':

    # Load data
	FTRAINF = '/data3/linhlv/pronotum/v1/csv/train_v19.csv'
	FTESTF = '/data3/linhlv/pronotum/v1/csv/test_v19.csv'
	X1,y1 = load2d(FTRAINF,test=False)

	#=================================================================
	# Load the parameters into list of layer, create a new network and train		
	newlayers = set_weights(FMODEL)
	net2 = build_fine_tuning_model(newlayers)
	net2.fit(X1,y1)
	
    # Save the fine-tuning model
	sys.setrecursionlimit(150000)
	with open('/data3/linhlv/2018/saveModels/cnnmodel_all_10000_pronotum_fine_tune_v19.pickle','wb') as f:
		pickle.dump(net2,f,-1)

	# draw the loss
	draw_loss(net2)

	# test the fine-tuning network and draw the results
	X, _ = load2d(FTESTF,test=True)
	y_pred = net2.predict(X)

	fig = pyplot.figure(figsize=(4, 4))
	fig.subplots_adjust(
    		left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
	for i in range(16):
    		ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    		plot_sample(X[i], y_pred[i], ax)
	pyplot.show()
	
	
	
