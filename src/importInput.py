import os
import numpy as np
from keras.utils import np_utils
from params import *
import wave
import pylab as plt
from keras.datasets import mnist
import pickle

def importInput():
    
    ### Load audio ###
    
    data_dir = "../data/oneSpeaker/"
    
    maxLenOfInputAudio = 0.5;#[s]
    
    #init dataset
    nCategory = 10
    nVariation = 50
    nSpeaker = 1
    
    yTrainAudio_orig = np.zeros(nCategory*nVariation*nSpeaker)
    # xTrainAudio_orig = np.zeros((nCategory*nVariation*nSpeaker,65,31))
    xTrainAudio_orig = np.zeros((nCategory*nVariation*nSpeaker,dim,dim))
    
    print("** loading input **");
    
    counter = 0;
    fileList = os.listdir(data_dir)
    for file_name in fileList:#["0_jackson_0.wav"]:#    
    #     print(file_name)
        wf = wave.open(data_dir+file_name,"rb")
        
        x = wf.readframes(wf.getnframes())
        x = np.frombuffer(x, dtype= "int16")
        copyLen = min(maxLenOfInputAudio,len(x))
        
        
        length = float(wf.getnframes()) / wf.getframerate()
        
        N = 128#256#512
        
        hammingWindow = np.hamming(N)
    
        maxNFrame = int(maxLenOfInputAudio * wf.getframerate());
        tmp = np.zeros(maxNFrame);
        copyLen = min(len(x),maxNFrame);
        tmp[:copyLen] = x[:copyLen];
        
        wf.close();
        
    
#         plt.subplot(1,2,1)
    #     plt.plot(tmp)
    
        
    #     plt.subplot(1,2,2)
        pxx, freqs, bins, im = plt.specgram(tmp, NFFT=N, Fs=wf.getframerate(), noverlap=0, window=hammingWindow)
            
    #     plt.axis([0, maxLenOfInputAudio, 0, wf.getframerate() / 2])
    #     plt.xlabel("time [second]")
    #     plt.ylabel("frequency [Hz]")
        
    #     plt.imshow(pxx[:dim,:dim])
    #     plt.show()
        
        xTrainAudio_orig[counter] = pxx[:dim,:dim];
        
        yTrainAudio_orig[counter] = int(file_name[0]); 
        counter=counter+1
    
    #normalise    
    xTrainAudio = xTrainAudio_orig / np.max(xTrainAudio_orig);
        
    #convert y to one-hot
    yTrainAudio = np_utils.to_categorical(yTrainAudio_orig)
    
    
    # # shuffling
    shuffleOrder = np.random.permutation(nCategory*nVariation*nSpeaker);
    xTrainAudio_shuffled = xTrainAudio[shuffleOrder];
    yTrainAudio_shuffled = yTrainAudio[shuffleOrder];
    
    
    # testing set (simply ordered)
    xTestAudio = xTrainAudio;
    yTestAudio = yTrainAudio;
    
    audioDataToSave = []
    audioDataToSave.append(xTrainAudio);
    audioDataToSave.append(yTrainAudio);
    audioDataToSave.append(xTrainAudio_shuffled);
    audioDataToSave.append(yTrainAudio_shuffled);
    audioDataToSave.append(xTestAudio);
    audioDataToSave.append(yTestAudio);
    
    
    output_audio = open('audioInput.pkl', 'wb')
    pickle.dump(audioDataToSave, output_audio)
    output_audio.close()
    
    print("** audio imported **")
    
    
    
    
    
    
    
    ### Load image ###

    
    (xTrainMnist_orig, yTrainMnist_orig), (xTestMnist_orig, yTestMnist_orig) = mnist.load_data()
    
    img_rows = 28;
    img_cols = 28;
    
    xTrainMnist_orig = xTrainMnist_orig.reshape(xTrainMnist_orig.shape[0], img_rows, img_cols).astype('float32')
    xTestMnist_orig = xTestMnist_orig.reshape(xTestMnist_orig.shape[0], img_rows, img_cols).astype('float32')
    
    # normalise
    xTrainMnist_shuffled = xTrainMnist_orig / 255
    xTestMnist_shuffled = xTestMnist_orig / 255
    
    #convert y to one-hot
    yTrainMnist_shuffled = np_utils.to_categorical(yTrainMnist_orig, num_classes=10)
    yTestMnist_shuffled = np_utils.to_categorical(yTestMnist_orig, num_classes=10)
    
    #generate ordered set
    nTestSet = 500;
    nTrainSet = nTestSet;
    xTestMnist = np.zeros((nTestSet,img_rows,img_rows));
    yTestMnist = np.zeros((nTestSet));
    xTrainMnist = np.zeros((nTrainSet,img_rows,img_rows));
    yTrainMnist = np.zeros((nTrainSet));
    for i in range(10):
        tmp = xTestMnist_shuffled[(yTestMnist_orig==i)]#take elements with a particular y value
        xTestMnist[i*50:(i+1)*50,:28,:28] = tmp[:50]#take first 50 elements where the output y should be i
        yTestMnist[i*50:(i+1)*50] = i;
        tmp = xTrainMnist_shuffled[(yTrainMnist_orig==i)]#take elements with a particular y value
        xTrainMnist[i*50:(i+1)*50,:28,:28] = tmp[:50]
        yTrainMnist[i*50:(i+1)*50] = i;
    
    # #convert y to one-hot
    yTestMnist = np_utils.to_categorical(yTestMnist, num_classes=10)
    yTrainMnist = np_utils.to_categorical(yTrainMnist, num_classes=10)
    
    
    
    
    visualDataToSave = []
    visualDataToSave.append(xTrainMnist);
    visualDataToSave.append(yTrainMnist);
    visualDataToSave.append(xTrainMnist_shuffled);
    visualDataToSave.append(yTrainMnist_shuffled);
    visualDataToSave.append(xTestMnist);
    visualDataToSave.append(yTestMnist);
    
    
    output_visual = open('visualInput.pkl', 'wb')
    pickle.dump(visualDataToSave, output_visual)
    output_visual.close()
    
    print("** audio imported **")
    
    
    
    
    print("** MNIST loaded **")
    
    
