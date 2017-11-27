from params import *
# from importInput import *
import pickle

import importInput
import plotting
import analysis


## Initialization ##

## import input data and save into a file ##
# importInput.importInput();


## load Inputs ##
pkl_file = open('audioInput.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close();
xTrainAudio = data[0];
yTrainAudio = data[1];
xTrainAudio_shuffled = data[2];
yTrainAudio_shuffled = data[3];
xTestAudio = data[4];
yTestAudio = data[5];
print("** audio input loaded **")

pkl_file = open('visualInput.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close();
xTrainVisual = data[0];
yTrainVisual = data[1];
xTrainVisual_shuffled = data[2];
yTrainVisual_shuffled = data[3];
xTestVisual = data[4];
yTestVisual = data[5];
print("** visual input loaded **")




### mod input
nTestSet = len(xTrainAudio);
nTrainSet = nTestSet;
emptyInput = np.zeros((nTrainSet,dim,dim));
 
#create a combined input to test combinations of A and V (V only, A only, and V+A)
xTrainVisual_comb = np.zeros((nTrainSet*3,dim,dim));
xTrainVisual_comb[:nTrainSet] = xTrainVisual
xTrainVisual_comb[nTrainSet:nTrainSet*2] = emptyInput
xTrainVisual_comb[nTrainSet*2:] = xTrainVisual
 
xTrainAudio_comb = np.zeros((nTrainSet*3,dim,dim));
xTrainAudio_comb[:nTrainSet] = emptyInput
xTrainAudio_comb[nTrainSet:nTrainSet*2] = xTrainAudio
xTrainAudio_comb[nTrainSet*2:] = xTrainAudio

# yTest (all on)
# When testing the reconstructability, both Mnist and Audio should be fully reconstructed
xTestVisual_comb_allon=np.zeros((nTestSet*3,dim,dim));
xTestVisual_comb_allon[:nTestSet] = xTrainVisual
xTestVisual_comb_allon[nTestSet:nTestSet*2] = xTrainVisual
xTestVisual_comb_allon[nTestSet*2:] = xTrainVisual
 
xTestAudio_comb_allon=np.zeros((nTestSet*3,dim,dim));
xTestAudio_comb_allon[:nTestSet] = xTrainAudio
xTestAudio_comb_allon[nTestSet:nTestSet*2] = xTrainAudio
xTestAudio_comb_allon[nTestSet*2:] = xTrainAudio

# shuffling
shuffleOrder = np.random.permutation(nTrainSet);
xTrainVisual_comb_shuffled = xTrainVisual_comb[np.concatenate((shuffleOrder,shuffleOrder+nTestSet,shuffleOrder+nTestSet*2))]
xTrainAudio_comb_shuffled = xTrainAudio_comb[np.concatenate((shuffleOrder,shuffleOrder+nTestSet,shuffleOrder+nTestSet*2))]
# yTrain_comb_shuffled = yTrain_comb[np.concatenate((shuffleOrder,shuffleOrder+nTestSet,shuffleOrder+nTestSet*2))]
 
print("** created: xTrainMnist_comb, xTrainAudio_comb, xTestMnist_comb_allon, xTrainMnist_comb_shuffled, xTrainAudio_comb_shuffled **")



## load model ##
import models
model_full, model_partial = models.model1();
# model_full, model_partial = models.model3_oneInput();
model_full.compile(optimizer='adadelta', loss='binary_crossentropy')
model_partial.compile(optimizer='adadelta', loss='binary_crossentropy')
untrainedWeights_full = model_full.get_weights();
untrainedWeights_partial = model_partial.get_weights();

print("** model is loaded and compiled")

## training the model ##
# model_full.fit([xTrainVisual_comb, xTrainAudio_comb], [xTestVisual_comb_allon, xTestAudio_comb_allon],
#                 epochs=1000,
#                 batch_size=10,
#                 shuffle=True,
#                 validation_data=([emptyInput, xTrainAudio], [xTrainVisual, xTrainAudio]))
# model_full.save_weights('autoencoder_comb_3000it.h5')


## loading trained weights from the previously trained model 
model_full.load_weights('autoencoder_comb_3000it.h5')
# model_full.load_weights('171122_autoencoder_comb_sigmoid.h5')
# model_full.load_weights('171122_autoencoder_oneLayer.h5');


print("** weights are set")
trainedWeights_full = model_full.get_weights();
trainedWeights_partial=model_partial.get_weights()

## plot results
# plotting.plotResults(model,emptyInput,xTestAudio);
# plotting.plotResults(model,xTestVisual,emptyInput);
# plotting.plotResults(model,xTestVisual,xTestAudio);


## analysis over the middle layer
model_partial.set_weights(untrainedWeights_partial);
predictedResult_untrained = model_partial.predict([xTrainVisual_comb, xTrainAudio_comb]);
# model_partial.set_weights(model_full.get_weights()[:len(model_partial.weights)]);
model_partial.set_weights(trainedWeights_partial);
predictedResult_trained = model_partial.predict([xTrainVisual_comb, xTrainAudio_comb]);


# info analysis for all V only, A only, V+A
nObj = 10;
nTrans = 50*3;
nRow = np.shape(predictedResult_untrained)[1];
nCol = np.shape(predictedResult_untrained)[2];
nDep = np.shape(predictedResult_untrained)[3];
  
results_reshaped_for_analysis_untrained = np.zeros((nObj,nTrans,nRow,nCol,nDep))
results_reshaped_for_analysis_trained = np.zeros((nObj,nTrans,nRow,nCol,nDep))
for s in range(nObj):
    results_reshaped_for_analysis_untrained[s,:50] = predictedResult_untrained[s*50:(s+1)*50]
    results_reshaped_for_analysis_untrained[s,50:100] = predictedResult_untrained[500+s*50:500+(s+1)*50]
    results_reshaped_for_analysis_untrained[s,100:150] = predictedResult_untrained[1000+s*50:1000+(s+1)*50]
  
    results_reshaped_for_analysis_trained[s,:50] = predictedResult_trained[s*50:(s+1)*50]
    results_reshaped_for_analysis_trained[s,50:100] = predictedResult_trained[500+s*50:500+(s+1)*50]
    results_reshaped_for_analysis_trained[s,100:150] = predictedResult_trained[1000+s*50:1000+(s+1)*50]
  
IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=False,nBins=3)
# IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True,thresholdMode=True)
# IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis_avg(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True)
# plotting.plotActivityOfCellsWithMaxInfo(IRs=IRs_weighted_list[1],results=results_reshaped_for_analysis_trained);


results_all_trained = results_reshaped_for_analysis_trained;

 
# # info analysis for all V only and A only
# nObj = 10;
# nTrans = 50*2;
# nRow = np.shape(predictedResult_untrained)[1];
# nCol = np.shape(predictedResult_untrained)[2];
# nDep = np.shape(predictedResult_untrained)[3];
#   
# results_reshaped_for_analysis_untrained = np.zeros((nObj,nTrans,nRow,nCol,nDep))
# results_reshaped_for_analysis_trained = np.zeros((nObj,nTrans,nRow,nCol,nDep))
# for s in range(nObj):
#     results_reshaped_for_analysis_untrained[s,:50] = predictedResult_untrained[s*50:(s+1)*50]
#     results_reshaped_for_analysis_untrained[s,50:100] = predictedResult_untrained[500+s*50:500+(s+1)*50]
#      
#     results_reshaped_for_analysis_trained[s,:50] = predictedResult_trained[s*50:(s+1)*50]
#     results_reshaped_for_analysis_trained[s,50:100] = predictedResult_trained[500+s*50:500+(s+1)*50]
     
# IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True,nBins=3)
# IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True,thresholdMode=True,threshold=0.6)
# IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis_avg(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True)
# plotting.plotActivityOfCellsWithMaxInfo(IRs=IRs_weighted_list[1],results=results_reshaped_for_analysis_trained,title="V and A");
# plotting.plotActivityOfCellsWithMaxInfo(IRs=IRs_list[1],results=results_reshaped_for_analysis_trained,title="V and A");






### info analysis (video only)
# reshape result to easily use single cell info analysis
nObj = 10;
nTrans = 50;
nRow = np.shape(predictedResult_untrained)[1];
nCol = np.shape(predictedResult_untrained)[2];
nDep = np.shape(predictedResult_untrained)[3];

results_reshaped_for_analysis_untrained = np.zeros((nObj,nTrans,nRow,nCol,nDep))
results_reshaped_for_analysis_trained = np.zeros((nObj,nTrans,nRow,nCol,nDep))

for s in range(nObj):
    results_reshaped_for_analysis_untrained[s,:50] = predictedResult_untrained[s*50:(s+1)*50]
    
    results_reshaped_for_analysis_trained[s,:50] = predictedResult_trained[s*50:(s+1)*50]
    

IRs_list_VOnly, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=False,nBins=3)
# IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True,thresholdMode=True)
# IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis_avg(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True)
# plotting.plotActivityOfCellsWithMaxInfo(IRs=IRs_weighted_list[1],results=results_reshaped_for_analysis_trained,title="V Only");
# plotting.plotActivityOfCellsWithMaxInfo(IRs=IRs_weighted_list[1],results=results_all_trained,title="based on info about V");









### info analysis (Audio only)
# reshape result to easily use single cell info analysis
nObj = 10;
nTrans = 50;
nRow = np.shape(predictedResult_untrained)[1];
nCol = np.shape(predictedResult_untrained)[2];
nDep = np.shape(predictedResult_untrained)[3];

results_reshaped_for_analysis_untrained = np.zeros((nObj,nTrans,nRow,nCol,nDep))
results_reshaped_for_analysis_trained = np.zeros((nObj,nTrans,nRow,nCol,nDep))

for s in range(nObj):
    results_reshaped_for_analysis_untrained[s,:50] = predictedResult_untrained[500+s*50:500+(s+1)*50]
    
    results_reshaped_for_analysis_trained[s,:50] = predictedResult_trained[500+s*50:500+(s+1)*50]
    

IRs_list_AOnly, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=False,nBins=3)
# IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True,thresholdMode=True)
# IRs_list, IRs_weighted_list_avg = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True)
# plotting.plotActivityOfCellsWithMaxInfo(IRs=IRs_weighted_list[1],results=results_reshaped_for_analysis_trained, title="A Only");
# plotting.plotActivityOfCellsWithMaxInfo(IRs=IRs_weighted_list[1],results=results_all_trained,title="based on info about A");


analysis.countCellsWithSelectivity(IRs_list_VOnly,IRs_list_AOnly,results_all_trained,plotOn=True);


