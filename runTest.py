try:
	import cPickle as pickle
except ImportError:
	import pickle
import os
import sys
import numpy as np
from lasagne import layers
from nolearn.lasagne import NeuralNet
from matplotlib import pyplot
from readCSV import loaddata, load2d
from utils import AdjustVariable, plot_sample, draw_loss, write_file
from pandas.io.parsers import read_csv
import theano


def loadCSV(fname = None):
    df = read_csv(os.path.expanduser(fname))
    df = df.dropna()
    imagePaths = df['Image']
    return imagePaths

def extract_fileNames(imagePaths):
    paths = imagePaths.values
    alist=[]
    print(len(alist))
    for i in range(len(paths)):
        pathi = paths[i]
        lastIndex = pathi.rfind('/')
        name = pathi[lastIndex+1:]
        alist.append(name)

    print(len(alist))
    return alist


FMODEL = '/data3/linhlv/OUTPUT2/2018/tete/fine_tuning/run_train_10landmarks/cnnmodel_10_output_fine_tuning_unfreeze_'
FTEST = '/home/linh/DATA/tete/v1/csv/test_'#v10.csv'
FSAVEFOLDER = '/data3/linhlv/OUTPUT2/2018/tete/fine_tuning/run_test_10landmarks/'
filename = FSAVEFOLDER + 'landmarks/cnnmodel_10_output_fine_tuning_unfreeze_'#.txt

FSAVEIMAGES = FSAVEFOLDER + 'images/'

DATA=['v10','v11','v12','v14','v15','v16','v17','v18','v19']
for i in DATA:
	fmodelf = FMODEL + i + '.pickle'
	ftestf = FTEST + i + '.csv'
	flandmarks = filename + i + '.txt'
	net = None
	sys.setrecursionlimit(100000)
	with open(fmodelf, 'rb') as f:
		net = pickle.load(f)

	X, _ = load2d(ftestf,test=True)
	y_pred = net.predict(X)

	# try to display the estimated landmarks on images
	paths = loadCSV(ftestf)
	fileNames = extract_fileNames(paths)

	for i in range(len(y_pred)):
		predi = y_pred[i]
		#filename = FSAVEFOLDER + FNAMES[i]
		write_file(flandmarks,predi)
		#write_file(filename,"\n")
		saveImg = FSAVEIMAGES + fileNames[i];
		fig = pyplot.figure()
		ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
		plot_sample(X[i],predi,ax)
		fig.savefig(saveImg)
		pyplot.close(fig)

	print('Finish!')


'''
# plot the landmarks on the images
fig = pyplot.figure(figsize=(4, 4))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(X[i], y_pred[i], ax)

pyplot.show()
'''
