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

from cnnmodel3_2018 import train
DATA=['v10','v11','v12','v14','v15','v16','v17','v18','v19']
FTRAIN_FIX = '/data3/linhlv/2018/csv/train_'
FTEST_FIX = '/data3/linhlv/2018/csv/test_'
SAVE_FIX = '/home/linh/Output/2018/pronotum_changed/3parts/cnnmodel_output_3parts_'
epochs = 5000
for i in DATA:
	ftrain = FTRAIN_FIX + i+'.csv'
	ftest = FTEST_FIX + i+'.csv'
	savemodel = SAVE_FIX + i+'.pickle'
	saveloss = SAVE_FIX + i+'_loss.jpg'
	savetest = SAVE_FIX + i+'_test.jpg'
	print(ftrain)
	print(ftest)
	print(savemodel)
	print(saveloss)
	print(savetest)
	train(ftrain,ftest,epochs,savemodel,saveloss,savetest)

print("Finish!!")
