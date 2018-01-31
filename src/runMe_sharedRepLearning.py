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

# # shuffling (this needs to be done at the same time for Visual inputs)
# shuffleOrder = np.random.permutation(nObj*nTrans);
# xTrainAudio_shuffled = xTrainAudio[shuffleOrder];
# yTrainAudio_shuffled = yTrainAudio[shuffleOrder];
# xTrainVisual_shuffled = xTrainVisual[shuffleOrder];
# yTrainVisual_shuffled = yTrainVisual[shuffleOrder];





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

# yTrain
yTrain_comb = np.zeros((nTrainSet*3,10));
yTrain_comb[:nTrainSet] = yTrainVisual_shuffled
yTrain_comb[nTrainSet:nTrainSet*2] = yTrainVisual_shuffled
yTrain_comb[nTrainSet*2:] = yTrainVisual_shuffled



# xTest (comb)
xTestVisual_comb = np.zeros((nTestSet*3,dim,dim));
xTestVisual_comb[:nTestSet] = xTestVisual
xTestVisual_comb[nTestSet:nTestSet*2] = emptyInput_visual_test
xTestVisual_comb[nTestSet*2:] = xTestVisual
 
xTestAudio_comb = np.zeros((nTestSet*3,int(dim/2),2*dim));
xTestAudio_comb[:nTestSet] = emptyInput_audio_test
xTestAudio_comb[nTestSet:nTestSet*2] = xTestAudio
xTestAudio_comb[nTestSet*2:] = xTestAudio

# yTest
yTest_comb = np.zeros((nTestSet*3,10));
yTest_comb[:nTestSet] = yTestVisual
yTest_comb[nTestSet:nTestSet*2] = yTestVisual
yTest_comb[nTestSet*2:] = yTestVisual




# shuffling
shuffleOrder = np.random.permutation(nTrainSet*3);

xTrainVisual_comb_shuffled = xTrainVisual_comb[shuffleOrder]
xTrainAudio_comb_shuffled = xTrainAudio_comb[shuffleOrder]

xTrainVisual_comb_allon_shuffled = xTrainVisual_comb_allon[shuffleOrder]
xTrainAudio_comb_allon_shuffled = xTrainAudio_comb_allon[shuffleOrder]

yTrain_comb_shuffled = yTrain_comb[shuffleOrder]
 
print("** created: xTrainMnist_comb, xTrainAudio_comb, xTestMnist_comb_allon, xTrainMnist_comb_shuffled, xTrainAudio_comb_shuffled **")









##########################
## Setting up the model ##
##########################

import models
# model_full, model_partial = models.model1();
# model_full, model_partial = models.model3_oneInput();
# model_full, model_partial = models.model4_dense();
# model_full, model_partial = models.model6_dense_4layers();
# model_full, model_partial = models.model7_dense_1layer();
model_full, model_partial = models.model8_revisedInput_4layer_reluDecoder_64();
# model_full, model_partial = models.model8_revisedInput_1layer_reluDecoder_64()#multi-layer decoder
# model_full, model_partial = models.model8_revisedInput_1layer_reluDecoder_64_decoderMod();#1layer decoder

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
model_full.load_weights('../data/171221_revisedStimuli_4layers_64_inconsistant_itr_0.weights');
# model_full.load_weights('../data/171221_revisedStimuli_4layers_64_consistant_itr_0.weights');
# model_full.load_weights('../data/171228_revisedStimuli_1layer_64_consistant_itr_0.weights');#multi-layer decoder
# model_full.load_weights('../data/170122_revisedStimuli_1layer_64_consistant_itr_0.weights');#1 layer decoder
untrainedWeights_full = model_full.get_weights();
untrainedWeights_partial = model_partial.get_weights();
 
 
# load trained weights
## loading trained weights from the previously trained model 
model_full.load_weights('../data/171221_revisedStimuli_4layers_64_inconsistant_itr_5000.weights');
# model_full.load_weights('../data/171221_revisedStimuli_4layers_64_consistant_itr_5000.weights');
# model_full.load_weights('../data/171228_revisedStimuli_1layer_64_consistant_itr_5000.weights');#multi-layer decoder
# model_full.load_weights('../data/170122_revisedStimuli_1layer_64_consistant_itr_5000.weights');#1 layer decoder
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




# experimentName = '180123_crossRepLearning_consit_A2V_1layer_supervised1layer';
# experimentName = '180123_crossRepLearning_consit_A2V_4layer_supervised1layer';
experimentName = '180125_crossRepLearning_consit_A2V_4layer_supervised3ayer_encodedInput';
# experimentName = '180123_crossRepLearning_consit_V2A_4layer_supervised1layer_encodedInput';

# model_crossRep = models.sharedRepLearning_decodedInput();
model_crossRep = models.sharedRepLearning_encodedInput();

# model_crossRep.compile(optimizer='adadelta', loss='binary_crossentropy',metrics=['categorical_accuracy'])
model_crossRep.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['categorical_accuracy'])
print("** model is loaded and compiled")
 
plot_model(model_crossRep, show_shapes=True, to_file='data/'+experimentName+'.png')
print("** model constructed **")

trainingItr = 0;
model_crossRep.save_weights('data/'+experimentName+'_itr_0.weights');
 
untrainedWeights_crossRep = model_crossRep.get_weights();
 
maxItr = 1000;
# #V2A
# model_crossRep.fit(predictedResult_crossRep_train_V, yTrainVisual_shuffled,
#                 epochs=maxItr,
#                 batch_size=256,
#                 validation_data=(predictedResult_crossRep_test_A, yTestAudio))



#A2V
model_crossRep.fit(predictedResult_crossRep_train_A, yTrainAudio_shuffled,
                epochs=maxItr,
                batch_size=256,
                validation_data=(predictedResult_crossRep_test_V, yTestVisual))
         

trainedWeights_crossRep = model_crossRep.get_weights();
model_crossRep.save_weights('data/'+experimentName+'_itr_'+str(maxItr)+'.weights');











## analysis over the middle layer
# predictedResult_crossRep_trained_test = model_crossRep.predict(predictedResult_crossRep_test_A);

# predict = np.argmax(model_crossRep.predict(predictedResult_crossRep_test_V),axis=1);
# print(sum(predict == np.argmax(yTestVisual,axis=1)) / (10*50))
# print('predict from V',predict)
# 
# 
# predict = np.argmax(model_crossRep.predict(predictedResult_crossRep_test_A),axis=1);
# print(sum(predict == np.argmax(yTestVisual,axis=1)) / (10*50))
# print('predict from A',predict)


score = model_crossRep.evaluate(predictedResult_crossRep_test_V, yTestVisual, verbose=0)
# print('Test loss (V): ', score[0])
print('Test acc. (V): ', score[1])
score = model_crossRep.evaluate(predictedResult_crossRep_test_A, yTestAudio, verbose=0)
# print('Test loss (A): ', score[0])
print('Test acc. (A): ', score[1])


# score = model_crossRep.evaluate(predictedResult_crossRep_train_V, yTrainVisual_shuffled, verbose=0)
# # print('Test loss (V): ', score[0])
# print('Test acc. (V): ', score[1])
# score = model_crossRep.evaluate(predictedResult_crossRep_train_A, yTrainAudio_shuffled, verbose=0)
# # print('Test loss (A): ', score[0])
# print('Test acc. (A): ', score[1])
