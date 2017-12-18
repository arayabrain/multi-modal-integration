from params import *
# from importInput import *
import pickle

import importInput
import plotting
import analysis


############################################
## import input data and save into a file ##
############################################
# importInput.importInput();



#################
## load Inputs ##
#################

pkl_file = open('data/audioInput.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close();
xTrainAudio = data[0];
yTrainAudio = data[1];
xTestAudio = data[2];
yTestAudio = data[3];
print("** audio input loaded **")

pkl_file = open('data/visualInput.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close();
xTrainVisual = data[0];
yTrainVisual = data[1];
xTestVisual = data[2];
yTestVisual = data[3];
print("** visual input loaded **")


nObj = 10;
nTrans = 50;#number of trans for loaded dataset

# shuffling (this needs to be done at the same time for Visual inputs)
shuffleOrder = np.random.permutation(nObj*nTrans);
xTrainAudio_shuffled = xTrainAudio[shuffleOrder];
yTrainAudio_shuffled = yTrainAudio[shuffleOrder];
xTrainVisual_shuffled = xTrainVisual[shuffleOrder];
yTrainVisual_shuffled = yTrainVisual[shuffleOrder];

# 
# #################################################
# ## modifying the structure of input to be used ##
# ## to avoid the network to learn a particular  ##
# ## pair of a visual input and a audio input    ##  
# #################################################
# 
nTestSet = len(xTestAudio);
nTrainSet = nTestSet;
emptyInput_visual = np.zeros((nTestSet,dim,dim));
emptyInput_audio = np.zeros((nTestSet,int(dim/2),dim*2));
# 
# 
# # param for training set

# nTransForEachCombination = 50*10;

# # combinations: V + A, V only, and A only
# # total number of presentation: (nTransForEachCombination*3)*nObj
# 
# xTrainV_comb = np.zeros((nObj*3*nTransForEachCombination,dim,dim));
# xTrainA_comb = np.zeros((nObj*3*nTransForEachCombination,dim,dim));
# xTrainV_allOn = np.zeros((nObj*3*nTransForEachCombination,dim,dim));
# xTrainA_allOn = np.zeros((nObj*3*nTransForEachCombination,dim,dim));
# 
# for o in range(nObj):
#     #V+A
#     shuffleOrder = (np.random.permutation(nTransForEachCombination)%nTrans) + o*nTrans;
#     xTrainV_comb[o*(nTransForEachCombination*3):o*(nTransForEachCombination*3)+nTransForEachCombination]=xTrainVisual[shuffleOrder];
#     xTrainV_allOn[o*(nTransForEachCombination*3):o*(nTransForEachCombination*3)+nTransForEachCombination]=xTrainVisual[shuffleOrder];
# 
#     shuffleOrder = (np.random.permutation(nTransForEachCombination)%nTrans) + o*nTrans;
#     xTrainA_comb[o*(nTransForEachCombination*3):o*(nTransForEachCombination*3)+nTransForEachCombination]=xTrainAudio[shuffleOrder];
#     xTrainA_allOn[o*(nTransForEachCombination*3):o*(nTransForEachCombination*3)+nTransForEachCombination]=xTrainAudio[shuffleOrder];
#     
#     #V only
#     shuffleOrder = (np.random.permutation(nTransForEachCombination)%nTrans) + o*nTrans;
#     xTrainV_comb[o*(nTransForEachCombination*3)+nTransForEachCombination:o*(nTransForEachCombination*3)+nTransForEachCombination*2]=xTrainVisual[shuffleOrder];
#     xTrainV_allOn[o*(nTransForEachCombination*3)+nTransForEachCombination:o*(nTransForEachCombination*3)+nTransForEachCombination*2]=xTrainVisual[shuffleOrder];
# 
#     xTrainA_comb[o*(nTransForEachCombination*3)+nTransForEachCombination:o*(nTransForEachCombination*3)+nTransForEachCombination*2]=emptyInput_visual[shuffleOrder];
#     xTrainA_allOn[o*(nTransForEachCombination*3)+nTransForEachCombination:o*(nTransForEachCombination*3)+nTransForEachCombination*2]=xTrainAudio[shuffleOrder];
#     
#     #A only
#     shuffleOrder = (np.random.permutation(nTransForEachCombination)%nTrans) + o*nTrans;
#     xTrainV_comb[o*(nTransForEachCombination*3)+nTransForEachCombination*2:o*(nTransForEachCombination*3)+nTransForEachCombination*3]=emptyInput_visual[shuffleOrder];
#     xTrainV_allOn[o*(nTransForEachCombination*3)+nTransForEachCombination*2:o*(nTransForEachCombination*3)+nTransForEachCombination*3]=xTrainVisual[shuffleOrder];
# 
#     xTrainA_comb[o*(nTransForEachCombination*3)+nTransForEachCombination*2:o*(nTransForEachCombination*3)+nTransForEachCombination*3]=xTrainAudio[shuffleOrder];
#     xTrainA_allOn[o*(nTransForEachCombination*3)+nTransForEachCombination*2:o*(nTransForEachCombination*3)+nTransForEachCombination*3]=xTrainAudio[shuffleOrder];
#  


#######################################################
## Creating a combined input to test combinations of ##
## A and V (V only, A only, and V+A)                 ##
#######################################################

xTrainVisual_comb = np.zeros((nTrainSet*3,dim,dim));
xTrainVisual_comb[:nTrainSet] = xTrainVisual_shuffled
xTrainVisual_comb[nTrainSet:nTrainSet*2] = emptyInput_visual
xTrainVisual_comb[nTrainSet*2:] = xTrainVisual_shuffled
 
xTrainAudio_comb = np.zeros((nTrainSet*3,int(dim/2),dim*2));
xTrainAudio_comb[:nTrainSet] = emptyInput_audio
xTrainAudio_comb[nTrainSet:nTrainSet*2] = xTrainAudio_shuffled
xTrainAudio_comb[nTrainSet*2:] = xTrainAudio_shuffled

# xTrain all on (all on)
# When testing the reconstructability, both Mnist and Audio should be fully reconstructed
xTrainVisual_comb_allon=np.zeros((nTrainSet*3,dim,dim));
xTrainVisual_comb_allon[:nTrainSet] = xTrainVisual_shuffled
xTrainVisual_comb_allon[nTrainSet:nTrainSet*2] = xTrainVisual_shuffled
xTrainVisual_comb_allon[nTrainSet*2:] = xTrainVisual_shuffled
 
xTrainAudio_comb_allon=np.zeros((nTrainSet*3,int(dim/2),dim*2));
xTrainAudio_comb_allon[:nTrainSet] = xTrainAudio_shuffled
xTrainAudio_comb_allon[nTrainSet:nTrainSet*2] = xTrainAudio_shuffled
xTrainAudio_comb_allon[nTrainSet*2:] = xTrainAudio_shuffled


# xTest (comb)
xTestVisual_comb = np.zeros((nTestSet*3,dim,dim));
xTestVisual_comb[:nTestSet] = xTestVisual
xTestVisual_comb[nTestSet:nTestSet*2] = emptyInput_visual
xTestVisual_comb[nTestSet*2:] = xTestVisual
 
xTestAudio_comb = np.zeros((nTestSet*3,int(dim/2),dim*2));
xTestAudio_comb[:nTestSet] = emptyInput_audio
xTestAudio_comb[nTestSet:nTestSet*2] = xTestAudio
xTestAudio_comb[nTestSet*2:] = xTestAudio



# shuffling
shuffleOrder = np.random.permutation(nTrainSet*3);

xTrainVisual_comb_shuffled = xTrainVisual_comb[shuffleOrder]
xTrainAudio_comb_shuffled = xTrainAudio_comb[shuffleOrder]

xTrainVisual_comb_allon_shuffled = xTrainVisual_comb_allon[shuffleOrder]
xTrainAudio_comb_allon_shuffled = xTrainAudio_comb_allon[shuffleOrder]

 
print("** created: xTrainMnist_comb, xTrainAudio_comb, xTestMnist_comb_allon, xTrainMnist_comb_shuffled, xTrainAudio_comb_shuffled **")




#######################################################
## Creating a combined input to test consistency of  ##
## V and A inputs.                                   ##
#######################################################
nVariations = nObj*nTrans;
xTestConsistency_V = np.zeros((nVariations*2,dim,dim));  
xTestConsistency_A = np.zeros((nVariations*2,dim,dim));  

# first half of the visual inputs are properly ordered, but the second half is randomly reordered
# importantly, the 

# xTestConsistency_V[:nVariations] = xTrainVisual
# 
# shuffleOrder = np.array((nObj*nTrans));
# for o in range(nObj):
#     shuffleOrder_tmp = np.random.permutation(nTrans-1);
#     if o!=nObj-1:
#         shuffleOrder_tmp[shuffleOrder_tmp==o]=nObj-1;
#     shuffleOrder[o*nTrans:(o+1)*nTrans]= shuffleOrder_tmp;
# 
# xTestConsistency_V[nVariations:nVariations*2] = xTrainVisual[shuffleOrder];
# xTestConsistency_A = np.concatenate((xTrainAudio,xTrainAudio));


    

# xTestConsistency_V[nVariations:nVariations*2] 






##########################
## Setting up the model ##
##########################

import models
# model_full, model_partial = models.model1();
# model_full, model_partial = models.model3_oneInput();
# model_full, model_partial = models.model4_dense();
# model_full, model_partial = models.model6_dense_4layers();
# model_full, model_partial = models.model7_dense_1layer();
model_full, model_partial = models.model8_revisedInput();

model_full.compile(optimizer='adadelta', loss='binary_crossentropy')
model_partial.compile(optimizer='adadelta', loss='binary_crossentropy')
# model_full.load_weights('data/171127_autoencoder_oneInput_itr_0.weights');
print("** model is loaded and compiled")



########################
## training the model ##
########################

# model_full.fit([xTrainVisual_comb, xTrainAudio_comb], [xTestVisual_comb_allon, xTestAudio_comb_allon],
#                 epochs=1000,
#                 batch_size=10,
#                 shuffle=True,
#                 validation_data=([emptyInput_visual, xTrainAudio], [xTrainVisual, xTrainAudio]))
# model_full.save_weights('autoencoder_comb_3000it.h5')





############################################
## Loading the weights from the save file ##
############################################

# load untrained weights
# model_full.load_weights('data/171128_autoencoder_dense_itr_0.weights');
model_full.load_weights('data/171218_revisedStimuli_1layer_consistant_itr_0.weights');
# model_full.load_weights('data/171130_model_dense_modTrain_itr_0.weights');
untrainedWeights_full = model_full.get_weights();
untrainedWeights_partial = model_partial.get_weights();


# load trained weights
## loading trained weights from the previously trained model 
# model_full.load_weights('data/autoencoder_comb_3000it.h5')
# model_full.load_weights('data/171122_autoencoder_comb_sigmoid.h5')
# model_full.load_weights('data/171127_autoencoder_oneInput_itr_5000.weights');
# model_full.load_weights('data/171128_autoencoder_dense_itr_10000.weights');
model_full.load_weights('data/171218_revisedStimuli_1layer_consistant_itr_1000.weights');
# model_full.load_weights('data/171130_model_dense_modTrain_itr_14000.weights');
# model_full.load_weights('data/171128_model_dense_4layers_itr_20000.weights');
trainedWeights_full = model_full.get_weights();
trainedWeights_partial=model_partial.get_weights()

print("** weights are set")




###############################
## plot reconstructed images ##
###############################

plotting.plotResults(model_full,xTestVisual,emptyInput_audio);
plotting.plotResults(model_full,emptyInput_visual,xTestAudio);
plotting.plotResults(model_full,xTestVisual,xTestAudio);






######################################
## Analysis:                        ##
## Retrieving the predicted results ##
######################################

## analysis over the middle layer
model_partial.set_weights(untrainedWeights_partial);
predictedResult_untrained = model_partial.predict([xTestVisual_comb, xTestAudio_comb]);
# model_partial.set_weights(model_full.get_weights()[:len(model_partial.weights)]);

model_partial.set_weights(trainedWeights_partial);
predictedResult_trained = model_partial.predict([xTestVisual_comb, xTestAudio_comb]);
















#################################
## Mutual information analysis ##
#################################

# ## calculate mutual info
# shape = np.shape(xTestVisual_comb);
# inputs_forMutual_V = xTestVisual_comb.reshape(shape[0],np.size(xTestVisual_comb[0]));
# inputs_forMutual_A = xTestAudio_comb.reshape(shape[0],np.size(xTestAudio_comb[0]));
#   
# shape = np.shape(predictedResult_trained);
# results_forMutual_trained = predictedResult_trained.reshape(shape[0],np.size(predictedResult_trained[0]));
# results_forMutual_untrained = predictedResult_untrained.reshape(shape[0],np.size(predictedResult_untrained[0]));
# IV_untrained = analysis.mutualInfo(inputs_forMutual_V,results_forMutual_untrained)
# IA_untrained = analysis.mutualInfo(inputs_forMutual_A,results_forMutual_untrained)
# IV_trained = analysis.mutualInfo(inputs_forMutual_V,results_forMutual_trained)
# IA_trained = analysis.mutualInfo(inputs_forMutual_A,results_forMutual_trained)
#   
#   
# print("** IV_untrained -- min: " + str(np.min(IV_untrained)) + ", max: " + str(np.max(IV_untrained)) + ", mean: " + str(np.mean(IV_untrained)));
# print("** IA_untrained -- min: " + str(np.min(IA_untrained)) + ", max: " + str(np.max(IA_untrained)) + ", mean: " + str(np.mean(IA_untrained)));
# print("** IV_Trained -- min: " + str(np.min(IV_trained)) + ", max: " + str(np.max(IV_trained)) + ", mean: " + str(np.mean(IV_trained)));
# print("** IA_Trained -- min: " + str(np.min(IA_trained)) + ", max: " + str(np.max(IA_trained)) + ", mean: " + str(np.mean(IA_trained)));
# 
# 
# ## display the info. table  
# plt.subplot(2,2,1);
# # plt.gray()
# im = plt.imshow(IV_untrained,vmin=0,vmax=1, aspect="auto");
# plt.colorbar(im)
# plt.title("Visual inputs (Untrained)");
# plt.xlabel("Encoded Units");
# plt.ylabel("Input Units")
# plt.subplot(2,2,2);
# im = plt.imshow(IA_untrained,vmin=0,vmax=1, aspect="auto");
# plt.colorbar(im)
# plt.title("Audio inputs (Untrained)");
# plt.xlabel("Encoded Units");
# plt.ylabel("Input Units")
#   
# plt.subplot(2,2,3)
# im = plt.imshow(IV_trained,vmin=0,vmax=1, aspect="auto");
# plt.colorbar(im)
# plt.title("Visual inputs (Trained)");
# plt.xlabel("Encoded Units");
# plt.ylabel("Input Units")
#   
# plt.subplot(2,2,4);
# im = plt.imshow(IA_trained,vmin=0,vmax=1, aspect="auto");
# plt.colorbar(im)
# plt.title("Audio inputs (Trained)");
# plt.xlabel("Encoded Units");
# plt.ylabel("Input Units")
# plt.show()
# 
# ## plot the info
# plt.subplot(2,1,1);
# plt.plot(-np.sort(-IV_trained.flatten()),label="IV_trained");
# plt.plot(-np.sort(-IV_untrained.flatten()),label="IV_untrained");
# plt.title("Visual Input Unit x Encoded Unit")
# plt.xlabel("s-r pair rank");
# plt.ylabel("Mutual Information [bit]")
# plt.ylim((-0.1,1.1));
# plt.legend();
#  
# plt.subplot(2,1,2);
# plt.plot(-np.sort(-IA_trained.flatten()),label="IA_trained");
# plt.plot(-np.sort(-IA_untrained.flatten()),label="IA_untrained");
# plt.title("Audio Input Unit x Encoded Unit")
# plt.xlabel("s-r pair rank");
# plt.ylabel("Mutual Information [bit]")
# plt.ylim((-0.1,1.1));
# plt.legend();
# plt.show()






######################################
## Single cell information analysis ##
######################################

## 1. info analysis for all V only, A only, V+A
nObj = 10;
nTrans = 50*3;
nRow = np.shape(predictedResult_untrained)[1];
nCol = np.shape(predictedResult_untrained)[2] if len(np.shape(predictedResult_untrained))>2 else 1;
nDep = np.shape(predictedResult_untrained)[3] if len(np.shape(predictedResult_untrained))>3 else 1;

if len(np.shape(predictedResult_untrained))<4:
    shape = np.shape(predictedResult_untrained);
    predictedResult_untrained = predictedResult_untrained.reshape((shape[0],nRow,nCol,nDep));
    predictedResult_trained = predictedResult_trained.reshape((shape[0],nRow,nCol,nDep));
  
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






## 2. info analysis based on Visual Inputs

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
    

IRs_list_VOnly, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True,nBins=3)
# IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True,thresholdMode=True)
# IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis_avg(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True)

# plotting.plotActivityOfCellsWithMaxInfo(IRs=IRs_weighted_list[1],results=results_all_trained,title="based on info about V");




## 3. info analysis based on Audio Inputs
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
    

IRs_list_AOnly, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True,nBins=3)
# IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True,thresholdMode=True)
# IRs_list, IRs_weighted_list_avg = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True)

# plotting.plotActivityOfCellsWithMaxInfo(IRs=IRs_weighted_list[1],results=results_all_trained,title="based on info about A");


## 4. show the stat of the single cell info analysis ##

analysis.countCellsWithSelectivity(IRs_list_VOnly,IRs_list_AOnly,results_all_trained,plotOn=True);



#############
## run PCA ##
#############

shape = np.shape(predictedResult_trained);
results_forPCA_trained = predictedResult_trained.reshape(shape[0],np.size(predictedResult_trained[0]));
results_forPCA_untrained = predictedResult_untrained.reshape(shape[0],np.size(predictedResult_untrained[0]));

## 1. PCA over stimulus category
analysis.runPCA(results_forPCA_untrained);
analysis.runPCA(results_forPCA_trained);

## 2. PCA over cells
# analysis.runPCAAboutUnits(np.transpose(results_forPCA_untrained),np.transpose(results_forPCA_trained),IRs_list_VOnly,IRs_list_AOnly);
# analysis.runPCAAboutUnits(np.transpose(results_forPCA_trained),IRs_list_VOnly[1],IRs_list_AOnly[1]);




