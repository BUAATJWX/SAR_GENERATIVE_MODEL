1.Operating environment
MATLAB R2016
Tensorflow 1.9.0
2.Please run ./loadData/loadMSTAR.m to load MSTAR raw files, and save the trainData.mat,trainAangle.mat and trainLabel.mat files to the ./data folder.
3.Please run ./run_main.py to train the model. When the training is finished, the generated images will be automatically saved in the fakeimdb_loss_*.mat,angle.mat,label.mat file.
4.Models trained with angles and labels can produce better quality images, and models trained using only labels can improve recognition rates.