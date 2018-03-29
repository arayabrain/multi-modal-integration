from params import *
# from importInput import *
import pickle
from keras.utils import plot_model

import importInput
import plotting
import analysis
import models


#################
## PARAMS      ##
#################

flag_useInconsistentTrainingDataset = False;

flag_trainingOn = False; #if true, then load the saved weights
maxItrForTraining = 5000;

flag_sharedRepLearning = True;
maxItrForSharedRepLearning = 1000;

flag_plotReconstructedImages = True;
flag_mutualInformationAnalysis = True;
flag_singleCellInfoAnalysis = True;
flag_plot_singleCell = True;
flag_PCA = True;


# experimentName = "171221_revisedStimuli_4layers_64_consistant";
experimentName = "4layers_64";
# experimentName = "4layers_64_Ngver";

if flag_useInconsistentTrainingDataset:
    experimentName = experimentName + "_inconsistent";
else:
    experimentName = experimentName + "_consistent";



##########################
## Setting up the model ##
##########################

## original 4 layer network
outputLayerOfPartialNet = 4;
model_full, model_partial = models.model_mixedInput_4Layers_64(outputLayerOfPartialNet=outputLayerOfPartialNet);

## different number of layers
# model_full, model_partial = models.model_mixedInput_3Layers_64();
# model_full, model_partial = models.model_mixedInput_2Layers_64();
# model_full, model_partial = models.model_mixedInput_1Layer_64();

## two-stage framework
# model_full, model_partial = models.model_twoStages_4Layers_64();



model_full.compile(optimizer='adadelta', loss='binary_crossentropy')
model_partial.compile(optimizer='adadelta', loss='binary_crossentropy')
# model_full.load_weights('data/171127_autoencoder_oneInput_itr_0.weights');
print("** model is loaded and compiled")



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

### CREATING INCONSISTENT PAIRS
if flag_useInconsistentTrainingDataset:
    shuffleOrder = np.random.permutation(np.shape(xTrainAudio_shuffled)[0]);
    xTrainAudio_shuffled = xTrainAudio_shuffled[shuffleOrder];
    yTrainAudio_shuffled = yTrainAudio_shuffled[shuffleOrder];



#######################################################
## Creating a combined input to test combinations of ##
## A and V (V only, A only, and V+A)                 ##
#######################################################

nTestSet = len(xTestVisual);
nTrainSet = len(xTrainVisual_shuffled);

dim = 28;
emptyInput_visual_test = np.zeros((nTestSet,dim,dim));
emptyInput_audio_test = np.zeros((nTestSet,int(dim/2),int(dim*2)));
emptyInput_visual_train = np.zeros((nTrainSet,dim,dim));
emptyInput_audio_train = np.zeros((nTrainSet,int(dim/2),int(dim*2)));

# xTrain (comb)
xTrainVisual_comb = np.zeros((nTrainSet*3,dim,dim));
xTrainVisual_comb[:nTrainSet] = xTrainVisual_shuffled
xTrainVisual_comb[nTrainSet:nTrainSet*2] = emptyInput_visual_train
xTrainVisual_comb[nTrainSet*2:] = xTrainVisual_shuffled
 
xTrainAudio_comb = np.zeros((nTrainSet*3,int(dim/2),int(dim*2)));
xTrainAudio_comb[:nTrainSet] = emptyInput_audio_train
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
xTestVisual_comb[nTestSet:nTestSet*2] = emptyInput_visual_test
xTestVisual_comb[nTestSet*2:] = xTestVisual
 
xTestAudio_comb = np.zeros((nTestSet*3,int(dim/2),2*dim));
xTestAudio_comb[:nTestSet] = emptyInput_audio_test
xTestAudio_comb[nTestSet:nTestSet*2] = xTestAudio
xTestAudio_comb[nTestSet*2:] = xTestAudio


# shuffling the training order altogether
shuffleOrder2 = np.random.permutation(nTrainSet*3);
 
xTrainVisual_comb_shuffled = xTrainVisual_comb[shuffleOrder2]
xTrainAudio_comb_shuffled = xTrainAudio_comb[shuffleOrder2]
 
xTrainVisual_comb_allon_shuffled = xTrainVisual_comb_allon[shuffleOrder2]
xTrainAudio_comb_allon_shuffled = xTrainAudio_comb_allon[shuffleOrder2]

print("** created: xTrainMnist_comb, xTrainAudio_comb, xTestMnist_comb_allon, xTrainMnist_comb_shuffled, xTrainAudio_comb_shuffled **")











########################
## training the model ##
########################
if flag_trainingOn:
#     experimentName = '171220_revisedStimuli_3layers_consistant_full';
     
    plot_model(model_full, show_shapes=True, to_file='data/'+experimentName+'.png')
    print("** model constructed **")
    trainingItr = 0;
    model_full.save_weights('data/'+experimentName+'_itr_0.weights');
     
    untrainedWeights_full = model_full.get_weights();
    untrainedWeights_partial = model_partial.get_weights();
     
    phaseSize = min(100,maxItrForTraining);
    while(trainingItr<maxItrForTraining):
        print("** \x1b[31m"+str(trainingItr)+"/"+str(maxItrForTraining)+"\x1b[0m")
        model_full.fit([xTrainVisual_comb_shuffled, xTrainAudio_comb_shuffled], [xTrainVisual_comb_allon_shuffled, xTrainAudio_comb_allon_shuffled],
                        epochs=phaseSize,
                        batch_size=50,
                        shuffle=True)
    #                     validation_data=([emptyInput_visual, xTrainAudio_shuffled], [xTrainVisual_shuffled, xTrainAudio_shuffled]))
        trainingItr+=phaseSize;
        model_full.save_weights('data/'+experimentName+'_itr_'+str(trainingItr)+'.weights');
            
    trainedWeights_full = model_full.get_weights();
    trainedWeights_partial=model_partial.get_weights()



############################################
## Loading the weights from the save file ##
############################################
if not flag_trainingOn:
    ##############################
    # Loading UNTRAINED weights ##
    ##############################
    
    model_full.load_weights('data/'+experimentName+'_itr_0.weights');
    
    
    
    ## original 4 layered network
#     model_full.load_weights('../data/171221_revisedStimuli_4layers_64_consistant_itr_0.weights');
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
     
     
    ############################
    # Loading TRAINED weights ##
    ############################
    
    ## loading trained weights from the previously trained model 
    model_full.load_weights('data/'+experimentName+'_itr_'+str(maxItrForTraining)+'.weights');
    
    
    ## original 4 layered network
    # model_full.load_weights('../data/171221_revisedStimuli_4layers_64_consistant_itr_5000.weights');
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
if flag_plotReconstructedImages:
    plotting.plotResults(model_full,xTestVisual,emptyInput_audio_test);
    plotting.plotResults(model_full,emptyInput_visual_test,xTestAudio);
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
if flag_mutualInformationAnalysis:
    # # calculate mutual info
    shape = np.shape(xTestVisual);
    inputs_forMutual_V = xTestVisual.reshape(shape[0],np.size(xTestVisual[0]));
    inputs_forMutual_A = xTestAudio.reshape(shape[0],np.size(xTestAudio[0]));
                 
    # shape = np.shape(predictedResult_trained[1000:]);
            
    results_forMutual_V_trained = predictedResult_trained[:500].reshape(500,np.size(predictedResult_trained[0]));
    results_forMutual_V_untrained = predictedResult_untrained[:500].reshape(500,np.size(predictedResult_untrained[0]));
            
    results_forMutual_A_trained = predictedResult_trained[500:1000].reshape(500,np.size(predictedResult_trained[0]));
    results_forMutual_A_untrained = predictedResult_untrained[500:1000].reshape(500,np.size(predictedResult_untrained[0]));
         
    nBins = 10;
    
    print("** running mutual info analysis about IV untrained **")
    IV_untrained = analysis.mutualInfo(inputs_forMutual_V,results_forMutual_V_untrained,nBins=nBins)
    print("** running mutual info analysis about IA untrained **")
    IA_untrained = analysis.mutualInfo(inputs_forMutual_A,results_forMutual_A_untrained,nBins=nBins)
           
    print("** running mutual info analysis about IV trained **")
    IV_trained = analysis.mutualInfo(inputs_forMutual_V,results_forMutual_V_trained,nBins=nBins)
    print("** running mutual info analysis about IA trained **")
    IA_trained = analysis.mutualInfo(inputs_forMutual_A,results_forMutual_A_trained,nBins=nBins)
               
         
    mutualInfoToSave = []
    mutualInfoToSave.append(IV_untrained);
    mutualInfoToSave.append(IA_untrained);
    mutualInfoToSave.append(IV_trained);
    mutualInfoToSave.append(IA_trained);
          
          
    pkl_file = open('data/'+experimentName+'_mutualInfo_bin20_l'+str(outputLayerOfPartialNet)+'.pkl', 'wb')
    pickle.dump(mutualInfoToSave, pkl_file)
    pkl_file.close();
    print("** mutual info exported **")
  
      
    ## shuffling the input pattern
    inputs_forMutual_V_shuffled = np.copy(inputs_forMutual_V);
    inputs_forMutual_A_shuffled = np.copy(inputs_forMutual_A);
      
    nInputs = np.shape(inputs_forMutual_V)[0];
    inputSize = np.shape(inputs_forMutual_V)[1];
      
    for index,index_shuffled in zip(range(nInputs*inputSize),np.random.permutation(nInputs*inputSize)):
        obj = index%nInputs;
        pixel = int(np.floor(index/nInputs));
    #     print(str(obj),str(pixel))
        pixel_shuffled = int(np.floor(index_shuffled/nInputs));
        obj_shuffled = index_shuffled%nInputs;
        inputs_forMutual_V_shuffled[obj,pixel] = inputs_forMutual_V[obj_shuffled,pixel_shuffled];
        inputs_forMutual_A_shuffled[obj,pixel] = inputs_forMutual_A[obj_shuffled,pixel_shuffled];
      
    print("** running mutual info analysis about IV shuffled **")
    IV_shuffled = analysis.mutualInfo(inputs_forMutual_V_shuffled,results_forMutual_V_trained,nBins=nBins)
    print("** running mutual info analysis about IA shuffled **")
    IA_shuffled = analysis.mutualInfo(inputs_forMutual_A_shuffled,results_forMutual_A_trained,nBins=nBins)
     
     
    # ## shuffling the output patterns
    # results_forMutual_V_trained_shuffled = np.copy(results_forMutual_V_trained);
    # results_forMutual_A_trained_shuffled = np.copy(results_forMutual_A_trained);
    # 
    # nStims = np.shape(results_forMutual_V_trained)[0];
    # layerDim = np.shape(results_forMutual_V_trained)[1];
    # 
    # for index,index_shuffled in zip(range(nStims*layerDim),np.random.permutation(nStims*layerDim)):
    #     obj = index%nStims;
    #     cellIndex = int(np.floor(index/nStims));
    # #     print(str(obj),str(pixel))
    #     cellIndex_shuffled = int(np.floor(index_shuffled/nStims));
    #     obj_shuffled = index_shuffled%nStims;
    #     results_forMutual_V_trained_shuffled[obj,cellIndex] = results_forMutual_V_trained[obj_shuffled,cellIndex_shuffled];
    #     results_forMutual_A_trained_shuffled[obj,cellIndex] = results_forMutual_A_trained[obj_shuffled,cellIndex_shuffled];
    #  
    # 
    # IV_shuffled = analysis.mutualInfo(inputs_forMutual_V,results_forMutual_V_trained_shuffled,nBins=nBins)
    # IA_shuffled = analysis.mutualInfo(inputs_forMutual_A,results_forMutual_A_trained_shuffled,nBins=nBins)
     
       
    mutualInfoToSave = []
    mutualInfoToSave.append(IV_shuffled);
    mutualInfoToSave.append(IA_shuffled);
    pkl_file = open('data/'+experimentName+'_mutualInfo_bin20_shuffled_l'+str(outputLayerOfPartialNet)+'.pkl', 'wb')
    pickle.dump(mutualInfoToSave, pkl_file)
    pkl_file.close();
    print("** mutual info (shuffled) exported **")
     
     
    print("** IV_untrained -- sum: " + str(np.sum(IV_untrained)) + ", max: " + str(np.max(IV_untrained)) + ", mean: " + str(np.mean(IV_untrained)));
    print("** IA_untrained -- sum: " + str(np.sum(IA_untrained)) + ", max: " + str(np.max(IA_untrained)) + ", mean: " + str(np.mean(IA_untrained)));
    print("** IV_Trained -- sum: " + str(np.sum(IV_trained)) + ", max: " + str(np.max(IV_trained)) + ", mean: " + str(np.mean(IV_trained)));
    print("** IA_Trained -- sum: " + str(np.sum(IA_trained)) + ", max: " + str(np.max(IA_trained)) + ", mean: " + str(np.mean(IA_trained)));
    print("** IV_Trained_shuffled -- sum: " + str(np.sum(IV_shuffled)) + ", max: " + str(np.max(IV_shuffled)) + ", mean: " + str(np.mean(IV_shuffled)));
    print("** IA_Trained_shuffled -- sum: " + str(np.sum(IA_shuffled)) + ", max: " + str(np.max(IA_shuffled)) + ", mean: " + str(np.mean(IA_shuffled)));
    input("Press Enter to continue...") 
 
######################################
## Single cell information analysis ##
######################################
if flag_singleCellInfoAnalysis:
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
          
    print("** info analysis for all V only, A only, V+A")     
    IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=flag_plot_singleCell,nBins=10)
    # plotting.plotActivityOfCellsWithMaxInfo(IRs=IRs_weighted_list[1],results=results_reshaped_for_analysis_trained);
             
             
    results_all_trained = results_reshaped_for_analysis_trained;
    pkl_file = open('data/'+experimentName+'_singleCellInfo_all_l'+str(outputLayerOfPartialNet)+'.pkl', 'wb')
    pickle.dump(IRs_list, pkl_file)
    pkl_file.close();
    print("** single cell info (all) exported **")    
    
    
    ## to create shuffled result
    results_reshaped_for_analysis_untrained_shuffled = np.copy(results_reshaped_for_analysis_untrained);
    results_reshaped_for_analysis_trained_shuffled = np.copy(results_reshaped_for_analysis_trained);
       
    for index,index_shuffled in zip(range(nObj*nTrans),np.random.permutation(nObj*nTrans)):
        obj = index%nObj;
        trans = int(np.floor(index/nObj));
        trans_shuffled = int(np.floor(index_shuffled/nObj));
        obj_shuffled = index_shuffled%nObj;
        results_reshaped_for_analysis_untrained_shuffled[obj,trans] = results_reshaped_for_analysis_untrained[obj_shuffled,trans_shuffled];
        results_reshaped_for_analysis_trained_shuffled[obj,trans] = results_reshaped_for_analysis_trained[obj_shuffled,trans_shuffled];
     
    print("** info analysis for all V only, A only, V+A (shuffled)")  
    IRs_list_oneModalityAtTime_shuffled, IRs_weighted_list_oneModalityAtTime = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained_shuffled,results_reshaped_for_analysis_trained_shuffled,plotOn=flag_plot_singleCell,nBins=10)
    pkl_file = open('data/'+experimentName+'_singleCellInfo_all_shuffled_l'+str(outputLayerOfPartialNet)+'.pkl', 'wb')
    pickle.dump(IRs_list_oneModalityAtTime_shuffled, pkl_file)
    pkl_file.close();
    print("** single cell info shuffled exported **")    
    
       
    ## 2. info analysis based on Visual Inputs
    nTrans = 50;
    results_reshaped_for_analysis_untrained = np.zeros((nObj,nTrans,nRow,nCol,nDep))
    results_reshaped_for_analysis_trained = np.zeros((nObj,nTrans,nRow,nCol,nDep))
           
    for s in range(nObj):
        results_reshaped_for_analysis_untrained[s,:50] = predictedResult_untrained[s*50:(s+1)*50]
               
        results_reshaped_for_analysis_trained[s,:50] = predictedResult_trained[s*50:(s+1)*50]
          
    print("** info analysis based on Visual Inputs")       
    IRs_list_VOnly, IRs_weighted_list_VOnly = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=flag_plot_singleCell,nBins=10)
           
#     plotting.plotActivityOfCellsWithMaxInfo(IRs=IRs_weighted_list_VOnly[1],results=results_all_trained,title="based on info about V");
    pkl_file = open('data/'+experimentName+'_singleCellInfo_V-Only_l'+str(outputLayerOfPartialNet)+'.pkl', 'wb')
    pickle.dump(IRs_list_VOnly, pkl_file)
    pkl_file.close();
    print("** Single Cell Info (V-Only) exported. **")
    
    
    ## 3. info analysis based on Audio Inputs
    # reshape result to easily use single cell info analysis
    nTrans = 50;
    results_reshaped_for_analysis_untrained = np.zeros((nObj,nTrans,nRow,nCol,nDep))
    results_reshaped_for_analysis_trained = np.zeros((nObj,nTrans,nRow,nCol,nDep))
           
    for s in range(nObj):
        results_reshaped_for_analysis_untrained[s,:50] = predictedResult_untrained[500+s*50:500+(s+1)*50]
               
        results_reshaped_for_analysis_trained[s,:50] = predictedResult_trained[500+s*50:500+(s+1)*50]
               
    print("** info analysis based on Audio Inputs")   
    IRs_list_AOnly, IRs_weighted_list_AOnly = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=flag_plot_singleCell,nBins=10)
           
#     plotting.plotActivityOfCellsWithMaxInfo(IRs=IRs_weighted_list_AOnly[1],results=results_all_trained,title="based on info about A");
     
    pkl_file = open('data/'+experimentName+'_singleCellInfo_A-Only_l'+str(outputLayerOfPartialNet)+'.pkl', 'wb')
    pickle.dump(IRs_list_AOnly, pkl_file)
    pkl_file.close();
    print("** Single Cell Info (A-Only) exported. **")
         

    ## 4. show the stat of the single cell info analysis ##
         
    analysis.countCellsWithSelectivity(IRs_list_VOnly,IRs_list_AOnly,results_all_trained,plotOn=True,infoThreshold = 1.0);
    input("Press Enter to continue...")





 
#############
## run PCA ##
#############
if flag_PCA:
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
 





######################################
## Analysis:                        ##
## Retrieving the predicted results ##
######################################
 
## analysis over the middle layer
# model_partial.set_weights(untrainedWeights_partial);
# predictedResult_untrained_train = model_partial.predict([xTrainVisual_shuffled, emptyInput_audio_train]);
# predictedResult_untrained_test = model_partial.predict([emptyInput_visual_test, xTestAudio]);


 


# model_full.set_weights(trainedWeights_full);
# predictedResult_crossRep_train_V = model_full.predict([xTrainVisual_shuffled, emptyInput_audio_train]);
# predictedResult_crossRep_test_V = model_full.predict([xTestVisual, emptyInput_audio_test]);
#  
# predictedResult_crossRep_train_A = model_full.predict([emptyInput_visual_train,xTrainAudio_shuffled]);
# predictedResult_crossRep_test_A = model_full.predict([emptyInput_visual_test,xTestAudio]);



model_partial.set_weights(trainedWeights_partial);
predictedResult_crossRep_train_V = model_partial.predict([xTrainVisual_shuffled, emptyInput_audio_train]);
predictedResult_crossRep_test_V = model_partial.predict([xTestVisual, emptyInput_audio_test]);
 
predictedResult_crossRep_train_A = model_partial.predict([emptyInput_visual_train,xTrainAudio_shuffled]);
predictedResult_crossRep_test_A = model_partial.predict([emptyInput_visual_test,xTestAudio]);







########################
## shared representation learning ##
########################
if flag_sharedRepLearning:
    
    
    ### REVERSING INCONSISTENT PAIRS
    if flag_useInconsistentTrainingDataset:
        xTrainAudio_shuffled[shuffleOrder] = xTrainAudio_shuffled;
        yTrainAudio_shuffled[shuffleOrder] = yTrainAudio_shuffled;
    
    #  
    # model_crossRep = models.sharedRepLearning();
    # 
    # model_crossRep.compile(optimizer='adadelta', loss='binary_crossentropy')
    # print("** model is loaded and compiled")
    #  
    # experimentName = '180119_crossRepLearning_consit';
    #  
    # plot_model(model_crossRep, show_shapes=True, to_file='data/'+experimentName+'.png')
    # print("** model constructed **")
    # 
    # trainingItr = 0;
    # model_crossRep.save_weights('data/'+experimentName+'_itr_0.weights');
    #  
    # untrainedWeights_crossRep = model_crossRep.get_weights();
    #  
    # phaseSize = 1;
    # maxItr = 1000;
    # while(trainingItr<=maxItr):
    #     print("** "+ str(trainingItr)+"/"+str(maxItr))
    #     model_crossRep.fit(predictedResult_crossRep_train_V, yTrainVisual_shuffled,
    #                     epochs=phaseSize,
    #                     batch_size=256,
    #                     shuffle=True)
    # #                     validation_data=([emptyInput_visual_test, xTestAudio], [xTestVisual, xTestAudio]))
    #     trainingItr+=phaseSize;
    # #     if trainingItr%10==0:
    # #         model_crossRep.save_weights('data/'+experimentName+'_itr_'+str(trainingItr*3)+'.weights');
    #          
    # trainedWeights_crossRep = model_crossRep.get_weights();
    
    
    
    
    # experimentName = '180125_crossRepLearning_consit_A2V_4layer_supervised3ayer_encodedInput';
    # experimentName = '180123_crossRepLearning_consit_V2A_4layer_supervised1layer_encodedInput';
    
    # experimentName = '180216_crossRepLearning_consit_A2V_4layers_supervised3ayer_encodedInput';
    # experimentName = '180220_crossRepLearning_consit_V2A_3layer_supervised3layer_encodedInput_ngver';
    
    
    # model_crossRep = models.sharedRepLearning_decodedInput();
    model_crossRep = models.sharedRepLearning_encodedInput();
    
    
    
    #########
    ## V2A ##
    #########
    
    print("*** Shared Rep Learning (V2A)");
    
    # model_crossRep.compile(optimizer='adadelta', loss='binary_crossentropy',metrics=['categorical_accuracy'])
    model_crossRep.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    print("** model is loaded and compiled")
     
#     plot_model(model_crossRep, show_shapes=True, to_file='data/'+experimentName+'_sharedRepLearning_V2A.png')
    
#     model_crossRep.save_weights('data/'+experimentName+'_sharedRepLearning_V2A_itr_0.weights');
     
#     untrainedWeights_crossRep = model_crossRep.get_weights();
     
    model_crossRep.fit(predictedResult_crossRep_train_V, yTrainVisual_shuffled,
                    epochs=maxItrForSharedRepLearning,
                    batch_size=256,
                    validation_data=(predictedResult_crossRep_test_A, yTestAudio))
    
    trainedWeights_crossRep = model_crossRep.get_weights();
    model_crossRep.save_weights('data/'+experimentName+'_sharedRepLearning_V2A_itr_'+str(maxItrForSharedRepLearning)+'.weights');
    
    
    score = model_crossRep.evaluate(predictedResult_crossRep_test_V, yTestVisual, verbose=0)
    # print('Test loss (V): ', score[0])
    print("** Result of Visual Training (V2A) **");
    print('Test acc. (V): ', score[1])
    score = model_crossRep.evaluate(predictedResult_crossRep_test_A, yTestAudio, verbose=0)
    # print('Test loss (A): ', score[0])
    print('Test acc. (A): ', score[1])
    input("Press Enter to continue...")
    
    
    
    
    #########
    ## A2V ##
    ######### 
    
    print("*** Shared Rep Learning (A2V)");
    
    model_crossRep.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    print("** model is loaded and compiled")
     
#     plot_model(model_crossRep, show_shapes=True, to_file='data/'+experimentName+'_sharedRepLearning_A2V.png')
    
#     model_crossRep.save_weights('data/'+experimentName+'_sharedRepLearning_A2V_itr_0.weights');
     
    untrainedWeights_crossRep = model_crossRep.get_weights();
     

    # #A2V
    model_crossRep.fit(predictedResult_crossRep_train_A, yTrainAudio_shuffled,
                    epochs=maxItrForSharedRepLearning,
                    batch_size=256,
                    validation_data=(predictedResult_crossRep_test_V, yTestVisual))
              
     
    trainedWeights_crossRep = model_crossRep.get_weights();
    model_crossRep.save_weights('data/'+experimentName+'_sharedRepLearning_A2V_itr_'+str(maxItrForSharedRepLearning)+'.weights');
    
    score = model_crossRep.evaluate(predictedResult_crossRep_test_V, yTestVisual, verbose=0)
    # print('Test loss (V): ', score[0])
    print("** Result of Auditory Training (A2V) **");
    print('Test acc. (V): ', score[1])
    score = model_crossRep.evaluate(predictedResult_crossRep_test_A, yTestAudio, verbose=0)
    # print('Test loss (A): ', score[0])
    print('Test acc. (A): ', score[1])


