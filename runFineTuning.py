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


from fineTune_freeze import fine_tune
'''
Fine-tuning the model v17 on 3 parts (pronotum, tete and elytre)

'''
#def fine_tune(fmodel,ftrain,epochs,ftest,savemodel,saveloss,savetest):

FMODEL = '/home/linh/Examples/trained_models/trained_Beetles/cnnmodel3_all_10000_epochs_.pickle'
DATA=['v10','v11','v12','v14','v15','v16','v17','v18','v19']
FTRAIN_FIX = '/home/linh/DATA/tete/v1/csv/train_'
FTEST_FIX = '/home/linh/DATA/tete/v1/csv/test_'
SAVE_FIX = '/data3/linhlv/train_frozen/cnnmodel_10_output_fine_tuning_freeze_'
epochs = 10000

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
	fine_tune(FMODEL,ftrain,epochs,ftest,savemodel,saveloss,savetest)

print("Finish!!")
