from params import *
# from importInput import *
import pickle
from keras.utils import plot_model

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

pkl_file = open('data/audioInput_5000.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close();
xTrainAudio_shuffled = data[0];
yTrainAudio_shuffled = data[1][:5000];
xTestAudio = data[2];
yTestAudio = data[3];
print("** audio input loaded **")

pkl_file = open('data/visualInput_5000.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close();
xTrainVisual_shuffled = data[0];
yTrainVisual_shuffled = data[1][:5000];
xTestVisual = data[2];
yTestVisual = data[3];
print("** visual input loaded **")


nObj = 10;
nTrans = 50;#number of trans for loaded dataset



#######################################################
## Creating a combined input to test combinations of ##
## A and V (V only, A only, and V+A)                 ##
#######################################################
nTestSet = len(xTestVisual);
nTrainSet = len(xTrainVisual_shuffled);

emptyInput_visual_test = np.zeros((nTestSet,dim,dim));
emptyInput_audio_test = np.zeros((nTestSet,int(dim/2),int(dim*2)));
emptyInput_visual_train = np.zeros((nTrainSet,dim,dim));
emptyInput_audio_train = np.zeros((nTrainSet,int(dim/2),int(dim*2)));


# xTrainVisual_comb = np.zeros((nTrainSet*3,dim,dim));
# xTrainVisual_comb[:nTrainSet] = xTrainVisual_shuffled
# xTrainVisual_comb[nTrainSet:nTrainSet*2] = emptyInput_visual_train
# xTrainVisual_comb[nTrainSet*2:] = xTrainVisual_shuffled
#  
# xTrainAudio_comb = np.zeros((nTrainSet*3,int(dim/2),dim*2));
# xTrainAudio_comb[:nTrainSet] = emptyInput_audio_train
# xTrainAudio_comb[nTrainSet:nTrainSet*2] = xTrainAudio_shuffled
# xTrainAudio_comb[nTrainSet*2:] = xTrainAudio_shuffled
# 
# # xTrain all on (all on)
# # When testing the reconstructability, both Mnist and Audio should be fully reconstructed
# xTrainVisual_comb_allon=np.zeros((nTrainSet*3,dim,dim));
# xTrainVisual_comb_allon[:nTrainSet] = xTrainVisual_shuffled
# xTrainVisual_comb_allon[nTrainSet:nTrainSet*2] = xTrainVisual_shuffled
# xTrainVisual_comb_allon[nTrainSet*2:] = xTrainVisual_shuffled
#  
# xTrainAudio_comb_allon=np.zeros((nTrainSet*3,int(dim/2),dim*2));
# xTrainAudio_comb_allon[:nTrainSet] = xTrainAudio_shuffled
# xTrainAudio_comb_allon[nTrainSet:nTrainSet*2] = xTrainAudio_shuffled
# xTrainAudio_comb_allon[nTrainSet*2:] = xTrainAudio_shuffled


# xTest (comb)
xTestVisual_comb = np.zeros((nTestSet*3,dim,dim));
xTestVisual_comb[:nTestSet] = xTestVisual
xTestVisual_comb[nTestSet:nTestSet*2] = emptyInput_visual_test
xTestVisual_comb[nTestSet*2:] = xTestVisual
 
xTestAudio_comb = np.zeros((nTestSet*3,int(dim/2),dim*2));
xTestAudio_comb[:nTestSet] = emptyInput_audio_test
xTestAudio_comb[nTestSet:nTestSet*2] = xTestAudio
xTestAudio_comb[nTestSet*2:] = xTestAudio


# 
# # shuffling
# shuffleOrder = np.random.permutation(nTrainSet*3);
# 
# xTrainVisual_comb_shuffled = xTrainVisual_comb[shuffleOrder]
# xTrainAudio_comb_shuffled = xTrainAudio_comb[shuffleOrder]
# 
# xTrainVisual_comb_allon_shuffled = xTrainVisual_comb_allon[shuffleOrder]
# xTrainAudio_comb_allon_shuffled = xTrainAudio_comb_allon[shuffleOrder]
# 
#  
print("** created: xTrainMnist_comb, xTrainAudio_comb, xTestMnist_comb_allon, xTrainMnist_comb_shuffled, xTrainAudio_comb_shuffled **")









##########################
## Setting up the model ##
##########################

import models

## original 4 layer network
outputLayerOfPartialNet = 4;
model_full, model_partial = models.model8_revisedInput_4layer_reluDecoder_64(outputLayerOfPartialNet=outputLayerOfPartialNet);

## different number of layers
# model_full, model_partial = models.model8_revisedInput_3layer_reluDecoder_64();
# model_full, model_partial = models.model8_revisedInput_2layer_reluDecoder_64();
# model_full, model_partial = models.model8_revisedInput_1layer_reluDecoder_64();

## two-stage framework
# model_full, model_partial = models.model9_separateInputs_3layer_reluDecoder_64();

# plot_model(model_full, show_shapes=True, to_file='check.png')


model_full.compile(optimizer='adadelta', loss='binary_crossentropy')
model_partial.compile(optimizer='adadelta', loss='binary_crossentropy')
# model_full.load_weights('data/171127_autoencoder_oneInput_itr_0.weights');
print("** model is loaded and compiled")



# ########################
# ## training the model ##
# ########################
# 
# experimentName = '171220_revisedStimuli_3layers_consistant_full';
# 
# plot_model(model_full, show_shapes=True, to_file='data/'+experimentName+'.png')
# print("** model constructed **")
# trainingItr = 0;
# model_full.save_weights('data/'+experimentName+'_itr_0.weights');
# 
# untrainedWeights_full = model_full.get_weights();
# untrainedWeights_partial = model_partial.get_weights();
# 
# phaseSize = 1;
# maxItr = 1000;
# while(trainingItr<=maxItr):
#     print("** "+ str(trainingItr)+"/"+str(maxItr))
#     model_full.fit([xTrainVisual_shuffled, xTrainAudio_shuffled], [xTrainVisual_shuffled, xTrainAudio_shuffled],
#                     epochs=phaseSize,
#                     batch_size=256,
#                     shuffle=True)
# #                     validation_data=([emptyInput_visual_test, xTestAudio], [xTestVisual, xTestAudio]))
#     model_full.fit([emptyInput_visual_train, xTrainAudio_shuffled], [xTrainVisual_shuffled, xTrainAudio_shuffled],
#                     epochs=phaseSize,
#                     batch_size=256,
#                     shuffle=True)
#     model_full.fit([xTrainVisual_shuffled, emptyInput_audio_train], [xTrainVisual_shuffled, xTrainAudio_shuffled],
#                     epochs=phaseSize,
#                     batch_size=256,
#                     shuffle=True)
#     trainingItr+=phaseSize;
#     if trainingItr%10==0:
#         model_full.save_weights('data/'+experimentName+'_itr_'+str(trainingItr*3)+'.weights');
#         
# trainedWeights_full = model_full.get_weights();
# trainedWeights_partial=model_partial.get_weights()





############################################
## Loading the weights from the save file ##
############################################
 
# load untrained weights
## original 4 layered network
model_full.load_weights('../data/171221_revisedStimuli_4layers_64_consistant_itr_0.weights');
# model_full.load_weights('../data/171221_revisedStimuli_4layers_64_inconsistant_itr_0.weights');
# model_full.load_weights('../data/171221_revisedStimuli_4layers_64_inconsistant_itr_5000.weights');

## two-stage framework (consistent)
# model_full.load_weights('../data/180209_revisedStimuli_4layers_64_consistant_Ngver_itr_0.weights');

## different number of layers (consistent)
# model_full.load_weights('../data/180122_revisedStimuli_1layer_64_consistant_itr_0.weights');
# model_full.load_weights('../data/180122_revisedStimuli_2layer_64_consistant_itr_0.weights');
# model_full.load_weights('../data/180122_revisedStimuli_3layer_64_consistant_itr_0.weights');

## different number of layers (inconsistent)
# model_full.load_weights('../data/189216_revisedStimuli_1layer_64_inconsistant_itr_0.weights');
# model_full.load_weights('../data/180216_revisedStimuli_2layer_64_inconsistant_itr_0.weights');
# model_full.load_weights('../data/180216_revisedStimuli_3layer_64_inconsistant_itr_0.weights');


untrainedWeights_full = model_full.get_weights();
untrainedWeights_partial = model_partial.get_weights();
 
 
# load trained weights
## loading trained weights from the previously trained model 

## original 4 layered network
model_full.load_weights('../data/171221_revisedStimuli_4layers_64_consistant_itr_5000.weights');
# model_full.load_weights('../data/171221_revisedStimuli_4layers_64_inconsistant_itr_5000.weights');

## two stage framework
# model_full.load_weights('../data/180209_revisedStimuli_4layers_64_consistant_Ngver_itr_5000.weights');

## different number of layers (consistent)
# model_full.load_weights('../data/180122_revisedStimuli_1layer_64_consistant_itr_5000.weights');
# model_full.load_weights('../data/180122_revisedStimuli_2layer_64_consistant_itr_5000.weights');
# model_full.load_weights('../data/180122_revisedStimuli_3layer_64_consistant_itr_5000.weights');

## different number of layers (inconsistent)
# model_full.load_weights('../data/189216_revisedStimuli_1layer_64_inconsistant_itr_5000.weights');
# model_full.load_weights('../data/180216_revisedStimuli_2layer_64_inconsistant_itr_5000.weights');
# model_full.load_weights('../data/180216_revisedStimuli_3layer_64_inconsistant_itr_5000.weights');

trainedWeights_full = model_full.get_weights();
trainedWeights_partial=model_partial.get_weights()
 
print("** weights are set")
 
 
 
 
###############################
## plot reconstructed images ##
###############################
 
# plotting.plotResults(model_full,xTestVisual,emptyInput_audio_test);
# plotting.plotResults(model_full,emptyInput_visual_test,xTestAudio);
# plotting.plotResults(model_full,xTestVisual,xTestAudio);
  
 
 
 
 
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
 
# print("untrained std: " + str(predictedResult_untrained.max()))
# print("trained std: " + str(predictedResult_trained.max()))
# 
# 
# 
# 
# # np.max(predictedResult_trained))
# yMax = 20000;
# plt.subplot(1,2,1)
# plt.hist(predictedResult_untrained.flatten(),20, range=( 0, 1))
# plt.xlim([0,1])
# plt.ylim([0,yMax*1.01])
# plt.title("Untrained Network")
# plt.subplot(1,2,2)
# plt.hist(predictedResult_trained.flatten(),20, range=( 0, 1))
# plt.xlim([0,1])
# plt.ylim([0,yMax*1.01])
# plt.title("Trained Network")
# plt.show()
 

 ###############################
 ## Persistant Homology
 ###############################
 
# analysis.persistentHomology(trainedWeights_full[0]);
 
 
 
 
#  #################################
#  ## Mutual information analysis ##
#  #################################
#     
# # # calculate mutual info
# shape = np.shape(xTestVisual);
# inputs_forMutual_V = xTestVisual.reshape(shape[0],np.size(xTestVisual[0]));
# inputs_forMutual_A = xTestAudio.reshape(shape[0],np.size(xTestAudio[0]));
#             
# # shape = np.shape(predictedResult_trained[1000:]);
#        
# results_forMutual_V_trained = predictedResult_trained[:500].reshape(500,np.size(predictedResult_trained[0]));
# results_forMutual_V_untrained = predictedResult_untrained[:500].reshape(500,np.size(predictedResult_untrained[0]));
#        
# results_forMutual_A_trained = predictedResult_trained[500:1000].reshape(500,np.size(predictedResult_trained[0]));
# results_forMutual_A_untrained = predictedResult_untrained[500:1000].reshape(500,np.size(predictedResult_untrained[0]));
#     
# nBins = 20;
# #  
# # IV_untrained = analysis.mutualInfo(inputs_forMutual_V,results_forMutual_V_untrained,nBins=nBins)
# # IA_untrained = analysis.mutualInfo(inputs_forMutual_A,results_forMutual_A_untrained,nBins=nBins)
# #       
# # IV_trained = analysis.mutualInfo(inputs_forMutual_V,results_forMutual_V_trained,nBins=nBins)
# # IA_trained = analysis.mutualInfo(inputs_forMutual_A,results_forMutual_A_trained,nBins=nBins)
# #           
# #     
# # mutualInfoToSave = []
# # mutualInfoToSave.append(IV_untrained);
# # mutualInfoToSave.append(IA_untrained);
# # mutualInfoToSave.append(IV_trained);
# # mutualInfoToSave.append(IA_trained);
# #      
# #      
# # pkl_file = open('data/mutualInfo_bin20_l'+str(outputLayerOfPartialNet)+'.pkl', 'wb')
# # pickle.dump(mutualInfoToSave, pkl_file)
# # pkl_file.close();
# # print("** mutual info exported **")
#  
#  
#  
#  
# ## shuffling the input pattern
# inputs_forMutual_V_shuffled = np.copy(inputs_forMutual_V);
# inputs_forMutual_A_shuffled = np.copy(inputs_forMutual_A);
#  
# nInputs = np.shape(inputs_forMutual_V)[0];
# inputSize = np.shape(inputs_forMutual_V)[1];
#  
# for index,index_shuffled in zip(range(nInputs*inputSize),np.random.permutation(nInputs*inputSize)):
#     obj = index%nInputs;
#     pixel = int(np.floor(index/nInputs));
# #     print(str(obj),str(pixel))
#     pixel_shuffled = int(np.floor(index_shuffled/nInputs));
#     obj_shuffled = index_shuffled%nInputs;
#     inputs_forMutual_V_shuffled[obj,pixel] = inputs_forMutual_V[obj_shuffled,pixel_shuffled];
#     inputs_forMutual_A_shuffled[obj,pixel] = inputs_forMutual_A[obj_shuffled,pixel_shuffled];
#  
# IV_shuffled = analysis.mutualInfo(inputs_forMutual_V_shuffled,results_forMutual_V_trained,nBins=nBins)
# IA_shuffled = analysis.mutualInfo(inputs_forMutual_A_shuffled,results_forMutual_A_trained,nBins=nBins)
# 
# 
# # ## shuffling the output patterns
# # results_forMutual_V_trained_shuffled = np.copy(results_forMutual_V_trained);
# # results_forMutual_A_trained_shuffled = np.copy(results_forMutual_A_trained);
# # 
# # nStims = np.shape(results_forMutual_V_trained)[0];
# # layerDim = np.shape(results_forMutual_V_trained)[1];
# # 
# # for index,index_shuffled in zip(range(nStims*layerDim),np.random.permutation(nStims*layerDim)):
# #     obj = index%nStims;
# #     cellIndex = int(np.floor(index/nStims));
# # #     print(str(obj),str(pixel))
# #     cellIndex_shuffled = int(np.floor(index_shuffled/nStims));
# #     obj_shuffled = index_shuffled%nStims;
# #     results_forMutual_V_trained_shuffled[obj,cellIndex] = results_forMutual_V_trained[obj_shuffled,cellIndex_shuffled];
# #     results_forMutual_A_trained_shuffled[obj,cellIndex] = results_forMutual_A_trained[obj_shuffled,cellIndex_shuffled];
# #  
# # 
# # IV_shuffled = analysis.mutualInfo(inputs_forMutual_V,results_forMutual_V_trained_shuffled,nBins=nBins)
# # IA_shuffled = analysis.mutualInfo(inputs_forMutual_A,results_forMutual_A_trained_shuffled,nBins=nBins)
# 
#   
# mutualInfoToSave = []
# mutualInfoToSave.append(IV_shuffled);
# mutualInfoToSave.append(IA_shuffled);
# pkl_file = open('data/mutualInfo_bin20_shuffled_l'+str(outputLayerOfPartialNet)+'.pkl', 'wb')
# pickle.dump(mutualInfoToSave, pkl_file)
# pkl_file.close();
# print("** mutual info exported **")
# 
# 
# # print("** IV_untrained -- sum: " + str(np.sum(IV_untrained)) + ", max: " + str(np.max(IV_untrained)) + ", mean: " + str(np.mean(IV_untrained)));
# # print("** IA_untrained -- sum: " + str(np.sum(IA_untrained)) + ", max: " + str(np.max(IA_untrained)) + ", mean: " + str(np.mean(IA_untrained)));
# # print("** IV_Trained -- sum: " + str(np.sum(IV_trained)) + ", max: " + str(np.max(IV_trained)) + ", mean: " + str(np.mean(IV_trained)));
# # print("** IA_Trained -- sum: " + str(np.sum(IA_trained)) + ", max: " + str(np.max(IA_trained)) + ", mean: " + str(np.mean(IA_trained)));
# print("** IV_Trained_shuffled -- sum: " + str(np.sum(IV_shuffled)) + ", max: " + str(np.max(IV_shuffled)) + ", mean: " + str(np.mean(IV_shuffled)));
# print("** IA_Trained_shuffled -- sum: " + str(np.sum(IA_shuffled)) + ", max: " + str(np.max(IA_shuffled)) + ", mean: " + str(np.mean(IA_shuffled)));


   
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
# # plt.ylim((max(IV_trained.max(),IV_untrained.max())*-0.1,max(IV_trained.max(),IV_untrained.max())*1.1));
# plt.ylim([-0.1,1.1])
# plt.legend();
#       
# plt.subplot(2,1,2);
# plt.plot(-np.sort(-IA_trained.flatten()),label="IA_trained");
# plt.plot(-np.sort(-IA_untrained.flatten()),label="IA_untrained");
# plt.title("Audio Input Unit x Encoded Unit")
# plt.xlabel("s-r pair rank");
# plt.ylabel("Mutual Information [bit]")
# # plt.ylim((max(IA_trained.max(),IA_untrained.max())*-0.1,max(IA_trained.max(),IA_untrained.max())*1.1));
# plt.ylim([-0.1,1.1])
# plt.legend();
# plt.subplots_adjust(hspace=0.5)
# plt.show()
 
 
 
 
 
 
# ######################################
# ## Single cell information analysis ##
# ######################################
nObj = 10;
nTrans = 50*3;
nRow = np.shape(predictedResult_untrained)[1];
nCol = np.shape(predictedResult_untrained)[2] if len(np.shape(predictedResult_untrained))>2 else 1;
nDep = np.shape(predictedResult_untrained)[3] if len(np.shape(predictedResult_untrained))>3 else 1;
      
if len(np.shape(predictedResult_untrained))<4:
    shape = np.shape(predictedResult_untrained);
    predictedResult_untrained = predictedResult_untrained.reshape((shape[0],nRow,nCol,nDep));
    predictedResult_trained = predictedResult_trained.reshape((shape[0],nRow,nCol,nDep));


## 1. info analysis for all V only, A only, V+A
results_reshaped_for_analysis_untrained = np.zeros((nObj,nTrans,nRow,nCol,nDep))
results_reshaped_for_analysis_trained = np.zeros((nObj,nTrans,nRow,nCol,nDep))
for s in range(nObj):
    results_reshaped_for_analysis_untrained[s,:50] = predictedResult_untrained[s*50:(s+1)*50]
    results_reshaped_for_analysis_untrained[s,50:100] = predictedResult_untrained[500+s*50:500+(s+1)*50]
    results_reshaped_for_analysis_untrained[s,100:150] = predictedResult_untrained[1000+s*50:1000+(s+1)*50]
           
    results_reshaped_for_analysis_trained[s,:50] = predictedResult_trained[s*50:(s+1)*50]
    results_reshaped_for_analysis_trained[s,50:100] = predictedResult_trained[500+s*50:500+(s+1)*50]
    results_reshaped_for_analysis_trained[s,100:150] = predictedResult_trained[1000+s*50:1000+(s+1)*50]
           
# IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=False,nBins=10)
# plotting.plotActivityOfCellsWithMaxInfo(IRs=IRs_weighted_list[1],results=results_reshaped_for_analysis_trained);
         
         
results_all_trained = results_reshaped_for_analysis_trained;
#       
# # pkl_file = open('data/singleCellInfo_consistent_bin10_l'+str(outputLayerOfPartialNet)+'.pkl', 'wb')
# # pickle.dump(IRs_list, pkl_file)
# # pkl_file.close();
# # print("** single cell info exported **")    
# 
# 
# ## to create shuffled result
# results_reshaped_for_analysis_untrained_shuffled = np.copy(results_reshaped_for_analysis_untrained);
# results_reshaped_for_analysis_trained_shuffled = np.copy(results_reshaped_for_analysis_trained);
#   
# for index,index_shuffled in zip(range(nObj*nTrans),np.random.permutation(nObj*nTrans)):
#     obj = index%nObj;
#     trans = int(np.floor(index/nObj));
#     print(str(obj),str(trans))
#     trans_shuffled = int(np.floor(index_shuffled/nObj));
#     obj_shuffled = index_shuffled%nObj;
#     results_reshaped_for_analysis_untrained_shuffled[obj,trans] = results_reshaped_for_analysis_untrained[obj_shuffled,trans_shuffled];
#     results_reshaped_for_analysis_trained_shuffled[obj,trans] = results_reshaped_for_analysis_trained[obj_shuffled,trans_shuffled];
# 
# 
#    
#    
# IRs_list_oneModalityAtTime_shuffled, IRs_weighted_list_oneModalityAtTime = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained_shuffled,results_reshaped_for_analysis_trained_shuffled,plotOn=False,nBins=10)
# pkl_file = open('data/singleCellInfo_consistent_shuffled_bin10_l'+str(outputLayerOfPartialNet)+'.pkl', 'wb')
# pickle.dump(IRs_list_oneModalityAtTime_shuffled, pkl_file)
# pkl_file.close();
# print("** single cell info shuffled exported **")    
     

   
## 2. info analysis based on Visual Inputs
nTrans = 50;
results_reshaped_for_analysis_untrained = np.zeros((nObj,nTrans,nRow,nCol,nDep))
results_reshaped_for_analysis_trained = np.zeros((nObj,nTrans,nRow,nCol,nDep))
       
for s in range(nObj):
    results_reshaped_for_analysis_untrained[s,:50] = predictedResult_untrained[s*50:(s+1)*50]
           
    results_reshaped_for_analysis_trained[s,:50] = predictedResult_trained[s*50:(s+1)*50]
           
# IRs_list_VOnly, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=False,nBins=3)
IRs_list_VOnly, IRs_weighted_list_VOnly = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=False,nBins=10)
# IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True,thresholdMode=True)
# IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis_avg(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True)
       
# plotting.plotActivityOfCellsWithMaxInfo(IRs=IRs_weighted_list_VOnly[1],results=results_all_trained,title="based on info about V");
# pkl_file = open('data/singleCellInfo_inconsistent_V-Only_bin10_l'+str(outputLayerOfPartialNet)+'.pkl', 'wb')
# pickle.dump(IRs_list_VOnly, pkl_file)
# pkl_file.close();


  
     
## 3. info analysis based on Audio Inputs
# reshape result to easily use single cell info analysis
# nObj = 10;
nTrans = 50;
# nRow = np.shape(predictedResult_untrained)[1];
# nCol = np.shape(predictedResult_untrained)[2];
# nDep = np.shape(predictedResult_untrained)[3];
       
results_reshaped_for_analysis_untrained = np.zeros((nObj,nTrans,nRow,nCol,nDep))
results_reshaped_for_analysis_trained = np.zeros((nObj,nTrans,nRow,nCol,nDep))
       
for s in range(nObj):
    results_reshaped_for_analysis_untrained[s,:50] = predictedResult_untrained[500+s*50:500+(s+1)*50]
           
    results_reshaped_for_analysis_trained[s,:50] = predictedResult_trained[500+s*50:500+(s+1)*50]
           
       
# IRs_list_AOnly, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=False,nBins=3)
IRs_list_AOnly, IRs_weighted_list_AOnly = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=False,nBins=10)
# IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True,thresholdMode=True)
# IRs_list, IRs_weighted_list_avg = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True)
       
# plotting.plotActivityOfCellsWithMaxInfo(IRs=IRs_weighted_list_AOnly[1],results=results_all_trained,title="based on info about A");
 
# pkl_file = open('data/singleCellInfo_inconsistent_A-Only_bin10_l'+str(outputLayerOfPartialNet)+'.pkl', 'wb')
# pickle.dump(IRs_list_AOnly, pkl_file)
# pkl_file.close();
     
     
     
     
# ## 4. info analysis based on one modality but concatenated
# nTrans = 100;
#             
# results_reshaped_for_analysis_untrained = np.zeros((nObj,nTrans,nRow,nCol,nDep))
# results_reshaped_for_analysis_trained = np.zeros((nObj,nTrans,nRow,nCol,nDep))
# for s in range(nObj):
#     results_reshaped_for_analysis_untrained[s,:50] = predictedResult_untrained[s*50:(s+1)*50]
#     results_reshaped_for_analysis_untrained[s,50:100] = predictedResult_untrained[500+s*50:500+(s+1)*50]
#         
#     results_reshaped_for_analysis_trained[s,:50] = predictedResult_trained[s*50:(s+1)*50]
#     results_reshaped_for_analysis_trained[s,50:100] = predictedResult_trained[500+s*50:500+(s+1)*50]
#           
# 
# results_all_trained = results_reshaped_for_analysis_trained;
# 
# # # IRs_list_AOnly, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=False,nBins=3)
# # IRs_list_oneModalityAtTime, IRs_weighted_list_oneModalityAtTime = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=False,nBins=10)
# # # IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True,thresholdMode=True)
# # # IRs_list, IRs_weighted_list_avg = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True)
# #       
# # # plotting.plotActivityOfCellsWithMaxInfo(IRs=IRs_weighted_list[1],results=results_all_trained,title="based on info about A");
# # pkl_file = open('data/singleCellInfo_consistent_oneModalityAtTime_bin10_l'+str(outputLayerOfPartialNet)+'.pkl', 'wb')
# # pickle.dump(IRs_list_oneModalityAtTime, pkl_file)
# # pkl_file.close();
# # print("** single cell info exported **")        
#      
#      
# # ## to create shuffled result
# # results_reshaped_for_analysis_untrained_shuffled = results_reshaped_for_analysis_untrained[np.random.permutation(nObj)]
# # results_reshaped_for_analysis_trained_shuffled = results_reshaped_for_analysis_trained[np.random.permutation(nObj)]
# # IRs_list_oneModalityAtTime_shuffled, IRs_weighted_list_oneModalityAtTime = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained_shuffled,results_reshaped_for_analysis_trained_shuffled,plotOn=False,nBins=10)
# # pkl_file = open('data/singleCellInfo_consistent_oneModalityAtTime_shuffled_bin10_l'+str(outputLayerOfPartialNet)+'.pkl', 'wb')
# # pickle.dump(IRs_list_oneModalityAtTime_shuffled, pkl_file)
# # pkl_file.close();
# # print("** single cell info shuffled exported **")       
     
     

## 4. show the stat of the single cell info analysis ##
     
analysis.countCellsWithSelectivity(IRs_list_VOnly,IRs_list_AOnly,results_all_trained,plotOn=True,infoThreshold = 1.0);
# analysis.countCellsWithSelectivity_or(IRs_list_VOnly,IRs_list_AOnly,results_all_trained,plotOn=True,infoThreshold = 1.3);


# analysis.countCellsWithSelectivity(IRs_weighted_list_VOnly,IRs_weighted_list_AOnly,results_all_trained,plotOn=True,infoThreshold = 1.0);
# 
#  






 
#############
## run PCA ##
#############
 
shape = np.shape(predictedResult_trained);
results_forPCA_untrained = predictedResult_untrained.reshape(shape[0],np.size(predictedResult_untrained[0]));
results_forPCA_trained = predictedResult_trained.reshape(shape[0],np.size(predictedResult_trained[0]));
        
## 1. PCA over stimulus category
analysis.runPCA(results_forPCA_untrained);
analysis.runPCA(results_forPCA_trained);
#  
## 2. PCA over cells
# analysis.runPCAAboutUnits(np.transpose(results_forPCA_untrained),np.transpose(results_forPCA_trained),IRs_list_VOnly,IRs_list_AOnly);
# analysis.runPCAAboutUnits(np.transpose(results_forPCA_trained),IRs_list_VOnly[1],IRs_list_AOnly[1]);
 

# ##############################
# ## Test Shared Rep Learning ##
# ##############################
# 
# predicted_fromV_trainingSet = model_partial.predict([xTrainVisual_shuffled, emptyInput_audio_train]);
# predicted_fromA_trainingSet = model_partial.predict([emptyInput_visual_train, xTrainAudio_shuffled]);
# 
# 
# visualInputOnly = predicted_fromV_trainingSet;
# visualInputTruth = yTrainVisual_shuffled;
# audioInputOnly = predicted_fromA_trainingSet;
# audioInputTruth = yTrainAudio_shuffled;
# 
# model_supervised=models.sharedRepLearning();
# 
# model_supervised.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
# print("** supervised model is loaded and compiled")
# 
# print("** shared representation learning with V only input");
# model_supervised.fit(visualInputOnly, visualInputTruth, batch_size=100, epochs=30, shuffle=True)
# 
# print("** predicting from A only input")
# audioInputOnly_testingSet = predictedResult_trained[500:1000];
# audioInputTruth_testingSet = yTestAudio;
# predict = np.argmax(model_supervised.predict(audioInputOnly_testingSet),axis=1);
# nCategory = 10;
# nVariation = 50;
# print(sum(predict == np.argmax(audioInputTruth_testingSet,axis=1)) / (nCategory*nVariation))
# print(predict)


