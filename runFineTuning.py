
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
