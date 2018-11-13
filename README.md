# CNN_Beetles_Landmarks
This repository contains the programs to create a CNN for landmarking on beetle's anatomical images.

The files and their meaning are following:
    - readCSV.py: load and normalize data. Normally, the data is stored in csv files.
    - utils.py: implements the helps functions such as drawing the losses, writing the file, or drawing the results, etc.
    - cnnmodel.py: defines the structure of the network and a method to train the network.
    - fineTune.py: contains the functions to load the trained model and fine-tuning it.
    - runTraining.py: this is the main file to train the model from scratch with cross-validation
    - runFineTuning.py: to fine-tune the trained model in different fold of data (cross-validation)
    - runTest.py: runs to predict the landmarks on the image of the test set.

# Notes:
    Contact to the author if you need the data to train/fine-tune a trained model.

# To train the network:
1. Change the location (file path) where you store the training/testing data (csv files)
2. Change the location where you would like to store the output model
3. Run the command: python runTraining.py

# To fine-tuning the trained model
1. Change the location to your trained model
2. Change the location of training/testing data (csv files)
3. Change the folder where you want to store the output model
4. Run the command: python runFineTuning.py

# To test a trained model on a test set
1. Change the file path to the trained model (pickle file)
2. Change the location of testing data
3. Change the folder where you want to store the output


