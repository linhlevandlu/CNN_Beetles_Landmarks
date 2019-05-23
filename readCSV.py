'''
 Elementary Blocks Network to landmark anatomical images
 Copyright (C) 2018  Le Van Linh (van-linh.le@u-bordeaux.fr)
 Version: 1.0
 Created on: March, 2018

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see http://www.gnu.org/licenses/.
'''


import os
import cv2
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

FTRAIN = '/data3/linhlv/2018/csv/train_v17.csv'
FTEST = '/data3/linhlv/2018/csv/test_v17.csv'

def readImage(arrayImgs):
	X = arrayImgs.values
	for i in range(len(X)):
		image = cv2.imread(X[i],cv2.IMREAD_GRAYSCALE)
		image = image.reshape(-1)
		image = image/255.
		image = image.astype(np.float32)
		X[i] = image
	X = np.vstack(X)
	return X

#scale the target to [-1,1]
def scaleTarget(target):
	print('Normalize target...')
	evencol = (target[:,::2] - 128)/128
	oddcol = (target[:,1::2] - 96)/96
	rs = np.empty((evencol.shape[0],evencol.shape[1] + oddcol.shape[1]))
	rs[:,::2] = evencol
	rs[:,1::2] = oddcol
	'''
	for i in range(len(target)):
		targeti = target[i]
		for j in range(len(targeti)):
			if j % 2 == 0:
				targeti[j] = (targeti[j] - 128)/128
			else:
				targeti[j] = (targeti[j] - 96)/96
		target[i] = targeti
	'''
	return rs

def loaddata(fname = None,test=False):
	if fname == None:
		fname = FTEST if test else FTRAIN
	df = read_csv(os.path.expanduser(fname))
	df = df.dropna()
	imagePath = df['Image']
	X = readImage(imagePath)
	if not test:
		y = df[df.columns[:-1]].values
		y = y.astype(np.float32)
		y = scaleTarget(y)
		X,y = shuffle(X,y,random_state=42)
		y = y.astype(np.float32)
		#print(y)
	else:
		y = None
	return X,y




# test loaddata method
#X,y = loaddata()
#print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
#    X.shape, X.min(), X.max()))
#print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
#    y.shape, y.min(), y.max()))

# reshape (convert) the data from 49152 to 192x256 (h x w)
def load2d(fname=None,test=False):
	print(fname)
	X,y = loaddata(fname,test=test)
	X = X.reshape(-1,1,192,256)
	if not test:
		print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(X.shape, X.min(), X.max()))
		print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(y.shape, y.min(), y.max()))
	return X,y
