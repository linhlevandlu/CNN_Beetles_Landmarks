
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
