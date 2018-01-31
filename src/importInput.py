import os
import numpy as np
from keras.utils import np_utils
from params import *
import wave
import pylab as plt
from keras.datasets import mnist
import pickle
from scipy.misc import imresize


def importInput2():
    ## use full MNIST training set
    ### Load audio ###
    
#     data_dir = "../data/oneSpeaker/"
    data_dir = "../data/twoSpeakers/"
    
    maxLenOfInputAudio = 1.0;#[s]
    
    #init dataset
    nCategory = 10
    nVariation = 50
    nSpeaker = 2
    
    
#     xLoadedAudio = np.zeros((nCategory*nVariation*nSpeaker,14,56))
#     yLoadedAudio = np.zeros(nCategory*nVariation*nSpeaker)
    # xLoadedAudio = np.zeros((nCategory*nVariation*nSpeaker,65,31))
    print("** Importing input **");
    
    
    xLoadedAudio = np.zeros((nCategory,nVariation*nSpeaker,int(dim/2),dim*2));
    
    counter = 0;
    counters = np.zeros((nCategory,));
    fileList = os.listdir(data_dir)
    for file_name in fileList:#["0_jackson_0.wav"]:#    
    #     print(file_name)
        wf = wave.open(data_dir+file_name,"rb")
        
        x = wf.readframes(wf.getnframes())
        x = np.frombuffer(x, dtype= "int16")
        copyLen = min(maxLenOfInputAudio,len(x))
        
        
        length = float(wf.getnframes()) / wf.getframerate()
        N = 148#256#512
        
        hammingWindow = np.hamming(N)
    
        maxNFrame = int(maxLenOfInputAudio * wf.getframerate());
        tmp = np.zeros(maxNFrame);
        copyLen = min(len(x),maxNFrame);
        tmp[:copyLen] = x[:copyLen];
        
        wf.close();
        
    
#         plt.subplot(1,2,1)
    #     plt.plot(tmp)
    
        
        plt.subplot(5,5,(counter%25)+1);
        pxx, freqs, bins, im = plt.specgram(tmp, NFFT=N, Fs=wf.getframerate(), noverlap=0, window=hammingWindow)
#         pxx, freqs, bins, im = plt.specgram(tmp)
#         plt.imshow(imresize(pxx[:15,:],(14, 56), interp='bilinear'),aspect='auto');
#         plt.title(file_name[0])
#         if ((counter+1)%25) == 0:
#             print(np.shape(imresize(pxx[:15,:],(14, 56), interp='bilinear')));
#             plt.show();
            
    #     plt.axis([0, maxLenOfInputAudio, 0, wf.getframerate() / 2])
    #     plt.xlabel("time [second]")
    #     plt.ylabel("frequency [Hz]")
        
    #     plt.imshow(pxx[:dim,:dim])
    #     plt.show()
        
        categoryIndex = int(file_name[0]);
        xLoadedAudio[categoryIndex,int(counters[categoryIndex]),:,:] = imresize(pxx[:15,:],(int(dim/2), dim*2), interp='bilinear');

        counter+=1;
        counters[categoryIndex]+=1;
    
#     plt.show();
    
    
    
    #normalise    
    xLoadedAudio_norm = xLoadedAudio / np.max(xLoadedAudio);
         
         
    # create test set (ordered), create subset of stimuli for training
    xLoadedAudio_trainSubSet = np.zeros((nCategory,nVariation,int(dim/2),dim*2));
    xTestAudio = np.zeros((nCategory*nVariation,int(dim/2),int(dim*2)));
    
    for obj in range(nCategory):
        shuffleOrder = np.random.permutation(nVariation*nSpeaker);
        xTestAudio[obj*nVariation:(obj+1)*nVariation] = xLoadedAudio_norm[obj,shuffleOrder[:nVariation]]
        xLoadedAudio_trainSubSet[obj,:] = xLoadedAudio_norm[obj,shuffleOrder[nVariation:]]
        
    
         
          
    ### Load image ###
    (xLoadedVisualTrain, yLoadedVisualTrain), (xLoadedVisualTest, yLoadedVisualTest) = mnist.load_data()
     
    img_rows = 28;
    img_cols = 28;
     
    xLoadedVisualTrain = xLoadedVisualTrain.reshape(xLoadedVisualTrain.shape[0], img_rows, img_cols).astype('float32')
    xLoadedVisualTest = xLoadedVisualTest.reshape(xLoadedVisualTest.shape[0], img_rows, img_cols).astype('float32')
     
    # normalise
    xLoadedVisualTrain = xLoadedVisualTrain / 255
    xLoadedVisualTest = xLoadedVisualTest / 255
     
 
    #generate ordered set
    nTestSet = 500;
    nTrainSet = 5000;
    xTestMnist = np.zeros((nTestSet,img_rows,img_rows));
    yTestMnist = np.zeros((nTestSet));
    for i in range(10):
        tmp = xLoadedVisualTest[(yLoadedVisualTest==i)]#take elements with a particular y value
        xTestMnist[i*50:(i+1)*50,:28,:28] = tmp[:50]#take first 50 elements where the output y should be i
        yTestMnist[i*50:(i+1)*50] = i;
        
        
        
    # create training set that match to the audio category
    xTrainMnist = xLoadedVisualTrain[:nTrainSet];
    sizeTrainingSet = np.shape(xTrainMnist)[0];
    
    
    xTrainAudio = np.zeros((sizeTrainingSet,int(dim/2),dim*2));
    
    # randomly assign corresponding spoken digit to xTrainAudio
    for i in range(sizeTrainingSet):
        number = yLoadedVisualTrain[i];
        transIndex = np.random.randint(nVariation);
        xTrainAudio[i] = xLoadedAudio_trainSubSet[number,transIndex];
        
    
    # #convert y to one-hot
    yTestMnist = np_utils.to_categorical(yTestMnist, num_classes=10)
    yTrainMnist = np_utils.to_categorical(yLoadedVisualTrain[:nTrainSet], num_classes=10)
    
    yTestAudio = yTestMnist;
    yTrainAudio = yTrainMnist;
     
     
     
    audioDataToSave = []
    audioDataToSave.append(xTrainAudio);
    audioDataToSave.append(yTrainAudio);
    audioDataToSave.append(xTestAudio);
    audioDataToSave.append(yTestAudio);
     
     
    output_audio = open('data/audioInput_5000.pkl', 'wb')
    pickle.dump(audioDataToSave, output_audio)
    output_audio.close()
    
     
    print("** audio imported **")
    
    
     
    visualDataToSave = []
    visualDataToSave.append(xTrainMnist);
    visualDataToSave.append(yTrainMnist);
    visualDataToSave.append(xTestMnist);
    visualDataToSave.append(yTestMnist);
     
     
    output_visual = open('data/visualInput_5000.pkl', 'wb')
    pickle.dump(visualDataToSave, output_visual)
    output_visual.close()
     

    print("** MNIST imported **")
#     






def importInput():
    
    ### Load audio ###
    
#     data_dir = "../data/oneSpeaker/"
    data_dir = "../data/twoSpeakers/"
    
    maxLenOfInputAudio = 1.0;#[s]
    
    #init dataset
    nCategory = 10
    nVariation = 50
    nSpeaker = 2
    
    
    xLoadedAudio = np.zeros((nCategory*nVariation*nSpeaker,14,56))
    yLoadedAudio = np.zeros(nCategory*nVariation*nSpeaker)
    # xLoadedAudio = np.zeros((nCategory*nVariation*nSpeaker,65,31))
    print("** Importing input **");
    
    counter = 0;
    fileList = os.listdir(data_dir)
    for file_name in fileList:#["0_jackson_0.wav"]:#    
    #     print(file_name)
        wf = wave.open(data_dir+file_name,"rb")
        
        x = wf.readframes(wf.getnframes())
        x = np.frombuffer(x, dtype= "int16")
        copyLen = min(maxLenOfInputAudio,len(x))
        
        
        length = float(wf.getnframes()) / wf.getframerate()
        N = 148#256#512
        
        hammingWindow = np.hamming(N)
    
        maxNFrame = int(maxLenOfInputAudio * wf.getframerate());
        tmp = np.zeros(maxNFrame);
        copyLen = min(len(x),maxNFrame);
        tmp[:copyLen] = x[:copyLen];
        
        wf.close();
        
    
#         plt.subplot(1,2,1)
    #     plt.plot(tmp)
    
        
        plt.subplot(5,5,(counter%25)+1);
        pxx, freqs, bins, im = plt.specgram(tmp, NFFT=N, Fs=wf.getframerate(), noverlap=0, window=hammingWindow)
#         pxx, freqs, bins, im = plt.specgram(tmp)
#         plt.imshow(imresize(pxx[:15,:],(14, 56), interp='bilinear'),aspect='auto');
#         plt.title(file_name[0])
#         if ((counter+1)%25) == 0:
#             print(np.shape(imresize(pxx[:15,:],(14, 56), interp='bilinear')));
#             plt.show();
            
    #     plt.axis([0, maxLenOfInputAudio, 0, wf.getframerate() / 2])
    #     plt.xlabel("time [second]")
    #     plt.ylabel("frequency [Hz]")
        
    #     plt.imshow(pxx[:dim,:dim])
    #     plt.show()
        
        xLoadedAudio[counter] = imresize(pxx[:15,:],(14, 56), interp='bilinear');
#         xLoadedAudio[counter] = pxx[:dim,:dim];
#         xLoadedAudio[counter] = imresize(pxx,(dim, dim), interp='bilinear')
        
        yLoadedAudio[counter] = int(file_name[0]); 
        

        counter=counter+1
    
#     plt.show();
    
    
    
    #normalise    
    xLoadedAudio_norm = xLoadedAudio / np.max(xLoadedAudio);
         
    #convert y to one-hot
    yLoadedAudio_onehot = np_utils.to_categorical(yLoadedAudio)
     
     
    xTestAudio = np.zeros((nCategory*nVariation,int(dim/2),int(dim*2)));
    xTrainAudio = np.zeros((nCategory*nVariation,int(dim/2),int(dim*2)));
    yTestAudio = np.zeros((nCategory*nVariation,nCategory));
    yTrainAudio = np.zeros((nCategory*nVariation,nCategory));
     
     
    # randomly choosing input for the training set and the testing set
    for obj in range(nCategory):
        shuffleOrder = np.random.permutation(nVariation*nSpeaker)+(obj*nVariation*nSpeaker);
        xTestAudio[obj*nVariation:(obj+1)*nVariation] = xLoadedAudio_norm[shuffleOrder[:nVariation]];
        yTestAudio[obj*nVariation:(obj+1)*nVariation] = yLoadedAudio_onehot[shuffleOrder[:nVariation]];
        xTrainAudio[obj*nVariation:(obj+1)*nVariation] = xLoadedAudio_norm[shuffleOrder[nVariation:]];
        yTrainAudio[obj*nVariation:(obj+1)*nVariation] = yLoadedAudio_onehot[shuffleOrder[nVariation:]];
         
        
     
     
#     # # shuffling (this needs to be done at the same time for Visual inputs)
#     shuffleOrder = np.random.permutation(nCategory*nVariation);
#     xTrainAudio_shuffled = xTrainAudio[shuffleOrder];
#     yTrainAudio_shuffled = yTrainAudio[shuffleOrder];
     
     
     
    audioDataToSave = []
    audioDataToSave.append(xTrainAudio);
    audioDataToSave.append(yTrainAudio);
    audioDataToSave.append(xTestAudio);
    audioDataToSave.append(yTestAudio);
     
     
    output_audio = open('data/audioInput.pkl', 'wb')
    pickle.dump(audioDataToSave, output_audio)
    output_audio.close()
     
    print("** audio imported **")
     
     
     
     
     
     
     
    ### Load image ###
 
     
    (xLoadedVisualTrain, yLoadedVisualTrain), (xLoadedVisualTest, yLoadedVisualTest) = mnist.load_data()
     
    img_rows = 28;
    img_cols = 28;
     
    xLoadedVisualTrain = xLoadedVisualTrain.reshape(xLoadedVisualTrain.shape[0], img_rows, img_cols).astype('float32')
    xLoadedVisualTest = xLoadedVisualTest.reshape(xLoadedVisualTest.shape[0], img_rows, img_cols).astype('float32')
     
    # normalise
    xLoadedVisualTrain = xLoadedVisualTrain / 255
    xLoadedVisualTest = xLoadedVisualTest / 255
     
 
     
    #generate ordered set
    nTestSet = 500;
    nTrainSet = nTestSet;
    xTestMnist = np.zeros((nTestSet,img_rows,img_rows));
    yTestMnist = np.zeros((nTestSet));
    xTrainMnist = np.zeros((nTrainSet,img_rows,img_rows));
    yTrainMnist = np.zeros((nTrainSet));
    for i in range(10):
        tmp = xLoadedVisualTest[(yLoadedVisualTest==i)]#take elements with a particular y value
        xTestMnist[i*50:(i+1)*50,:28,:28] = tmp[:50]#take first 50 elements where the output y should be i
        yTestMnist[i*50:(i+1)*50] = i;
        tmp = xLoadedVisualTrain[(yLoadedVisualTrain==i)]#take elements with a particular y value
        xTrainMnist[i*50:(i+1)*50,:28,:28] = tmp[:50]
        yTrainMnist[i*50:(i+1)*50] = i;
     
    # #convert y to one-hot
    yTestMnist = np_utils.to_categorical(yTestMnist, num_classes=10)
    yTrainMnist = np_utils.to_categorical(yTrainMnist, num_classes=10)
     
      #convert y to one-hot
    yLoadedVisualTrain = np_utils.to_categorical(yLoadedVisualTrain, num_classes=10)  
     
     
    visualDataToSave = []
    visualDataToSave.append(xTrainMnist);
    visualDataToSave.append(yTrainMnist);
    visualDataToSave.append(xTestMnist);
    visualDataToSave.append(yTestMnist);
     
     
    output_visual = open('data/visualInput.pkl', 'wb')
    pickle.dump(visualDataToSave, output_visual)
    output_visual.close()
     

    print("** MNIST imported **")
#     
    
