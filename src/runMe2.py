from params import *
# from importInput import *
import pickle

import importInput
import analysis
import plotting

## Autoencoder like thing with PCA


## Initialization ##

## import input data and save into a file ##
# importInput.importInput();

dim = 28;

## load Inputs ##
pkl_file = open('data/audioInput.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close();
xTrainAudio = data[0];
yTrainAudio = data[1];
xTrainAudio_shuffled = data[2];
yTrainAudio_shuffled = data[3];
xTestAudio = data[4];
yTestAudio = data[5];
print("** audio input loaded **")

pkl_file = open('data/visualInput.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close();
xTrainVisual = data[0];
yTrainVisual = data[1];
xTrainVisual_shuffled = data[2];
yTrainVisual_shuffled = data[3];
xTestVisual = data[4];
yTestVisual = data[5];
print("** visual input loaded **")


XVisual = np.reshape(xTrainVisual,(50*10,dim*dim))
XAudio = np.reshape(xTrainAudio,(50*10,dim*dim))

X = np.zeros((50*10,dim*dim*2));
X[:,:dim*dim] = XVisual;
X[:,dim*dim:dim*dim*2] = XAudio;

print(X.shape)


from sklearn.decomposition import PCA
N = 128
pca = PCA(n_components=N)
pca.fit(X)




# plot
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# cols = 10
# rows = int(np.ceil(N/float(cols)))
# 
# fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
# 
# for i in range(N):
#     r = i // cols
#     c = i % cols
#     axes[r, c].imshow(pca.components_[i].reshape(dim*2,dim),vmin=-0.5,vmax=0.5, cmap = cm.Greys_r)
#     axes[r, c].set_title('component %d' % i)
#     axes[r, c].get_xaxis().set_visible(False)
#     axes[r, c].get_yaxis().set_visible(False)




### mod input
nTestSet = len(xTrainAudio);
nTrainSet = nTestSet;
emptyInput = np.zeros((nTrainSet,dim*dim));
 
#create a combined input to test combinations of A and V (V only, A only, and V+A)
xTrain_comb = np.zeros((nTrainSet*3,dim*dim*2));
xTrain_comb[:nTrainSet,:dim*dim] = XVisual
xTrain_comb[nTrainSet:nTrainSet*2,:dim*dim] = emptyInput
xTrain_comb[nTrainSet*2:,:dim*dim] = XVisual
 
xTrain_comb[:nTrainSet,dim*dim:dim*dim*2] = emptyInput
xTrain_comb[nTrainSet:nTrainSet*2,dim*dim:dim*dim*2] = XAudio
xTrain_comb[nTrainSet*2:,dim*dim:dim*dim*2] = XAudio



xTrain_tmp = np.zeros((nTrainSet,dim*dim*2));
xTrain_tmp[:nTrainSet,:dim*dim] = XVisual
xTrain_tmp[:nTrainSet,dim*dim:dim*dim*2] = emptyInput;
# projection(V only)
predictedResult = pca.transform(xTrain_tmp)
# inverse
Xe = pca.inverse_transform(predictedResult)

fig, axes = plt.subplots(ncols=10, nrows=4, figsize=(30,4))

for i in range(10):
    #plot V
    axes[0, i].imshow(xTrain_tmp[i*50,:28*28].reshape(28,28),vmin=0.0,vmax=1.0, cmap = cm.Greys_r)
    axes[0, i].set_title('original %d' % i)
    axes[0, i].get_xaxis().set_visible(False)
    axes[0, i].get_yaxis().set_visible(False)
    
    #plot A
    axes[1, i].imshow(xTrain_tmp[i*50,28*28:28*28*2].reshape(28,28),cmap = cm.Greys_r)
    axes[1, i].set_title('original %d' % i)
    axes[1, i].get_xaxis().set_visible(False)
    axes[1, i].get_yaxis().set_visible(False)

    axes[2, i].imshow(Xe[i*50,:28*28].reshape(28,28),vmin=0.0,vmax=1.0, cmap = cm.Greys_r)
    axes[2, i].set_title('dimension reduction %d' % i)
    axes[2, i].get_xaxis().set_visible(False)
    axes[2, i].get_yaxis().set_visible(False)

    axes[3, i].imshow(Xe[i*50,28*28:28*28*2].reshape(28,28),cmap = cm.Greys_r)
    axes[3, i].set_title('dimension reduction %d' % i)
    axes[3, i].get_xaxis().set_visible(False)
    axes[3, i].get_yaxis().set_visible(False)
    
plt.show()




xTrain_tmp = np.zeros((nTrainSet,dim*dim*2));
xTrain_tmp[:nTrainSet,:dim*dim] = emptyInput
xTrain_tmp[:nTrainSet,dim*dim:dim*dim*2] = XAudio;
# projection(V only)
predictedResult = pca.transform(xTrain_tmp)
# inverse
Xe = pca.inverse_transform(predictedResult)

fig, axes = plt.subplots(ncols=10, nrows=4, figsize=(30,4))

for i in range(10):
    #plot V
    axes[0, i].imshow(xTrain_tmp[i*50,:28*28].reshape(28,28),vmin=0.0,vmax=1.0, cmap = cm.Greys_r)
    axes[0, i].set_title('original %d' % i)
    axes[0, i].get_xaxis().set_visible(False)
    axes[0, i].get_yaxis().set_visible(False)
    
    #plot A
    axes[1, i].imshow(xTrain_tmp[i*50,28*28:28*28*2].reshape(28,28),cmap = cm.Greys_r)
    axes[1, i].set_title('original %d' % i)
    axes[1, i].get_xaxis().set_visible(False)
    axes[1, i].get_yaxis().set_visible(False)

    axes[2, i].imshow(Xe[i*50,:28*28].reshape(28,28),vmin=0.0,vmax=1.0, cmap = cm.Greys_r)
    axes[2, i].set_title('dimension reduction %d' % i)
    axes[2, i].get_xaxis().set_visible(False)
    axes[2, i].get_yaxis().set_visible(False)

    axes[3, i].imshow(Xe[i*50,28*28:28*28*2].reshape(28,28),cmap = cm.Greys_r)
    axes[3, i].set_title('dimension reduction %d' % i)
    axes[3, i].get_xaxis().set_visible(False)
    axes[3, i].get_yaxis().set_visible(False)
    
plt.show()





xTrain_tmp = np.zeros((nTrainSet,dim*dim*2));
xTrain_tmp[:nTrainSet,:dim*dim] = XVisual
xTrain_tmp[:nTrainSet,dim*dim:dim*dim*2] = XAudio;
# projection(V only)
predictedResult = pca.transform(xTrain_tmp)
# inverse
Xe = pca.inverse_transform(predictedResult)

fig, axes = plt.subplots(ncols=10, nrows=4, figsize=(30,4))

for i in range(10):
    #plot V
    axes[0, i].imshow(xTrain_tmp[i*50,:28*28].reshape(28,28),vmin=0.0,vmax=1.0, cmap = cm.Greys_r)
    axes[0, i].set_title('original %d' % i)
    axes[0, i].get_xaxis().set_visible(False)
    axes[0, i].get_yaxis().set_visible(False)
    
    #plot A
    axes[1, i].imshow(xTrain_tmp[i*50,28*28:28*28*2].reshape(28,28),cmap = cm.Greys_r)
    axes[1, i].set_title('original %d' % i)
    axes[1, i].get_xaxis().set_visible(False)
    axes[1, i].get_yaxis().set_visible(False)

    axes[2, i].imshow(Xe[i*50,:28*28].reshape(28,28),vmin=0.0,vmax=1.0, cmap = cm.Greys_r)
    axes[2, i].set_title('dimension reduction %d' % i)
    axes[2, i].get_xaxis().set_visible(False)
    axes[2, i].get_yaxis().set_visible(False)

    axes[3, i].imshow(Xe[i*50,28*28:28*28*2].reshape(28,28),cmap = cm.Greys_r)
    axes[3, i].set_title('dimension reduction %d' % i)
    axes[3, i].get_xaxis().set_visible(False)
    axes[3, i].get_yaxis().set_visible(False)
    
plt.show()













# projection
# Xd = pca.transform(X)
predictedResult = pca.transform(xTrain_comb)
print(predictedResult.shape)




# #############
# ## run PCA ##
# #############
# 
# shape = np.shape(predictedResult);
# results_forPCA_trained = predictedResult.reshape(shape[0],np.size(predictedResult[0]));
# results_forPCA_untrained = predictedResult.reshape(shape[0],np.size(predictedResult[0]));
# 
# ## 1. PCA over stimulus category
# analysis.runPCA(results_forPCA_untrained);
# analysis.runPCA(results_forPCA_trained);






#s t row col dep
nCombPattern = 3;
nTrans = 50;
nStim = 10;
nRow = nCol = 4
nDep = 8;


# info analysis for all V only, A only, V+A
results_reshaped_for_analysis = np.zeros((nStim,nTrans*3,nRow,nCol,nDep));
varIndex = 0;
for s in range(nStim):
    for t in range(nTrans):
        cellIndex=0;
        for r in range(nRow):
            for c in range(nCol):
                for d in range(nDep):
                    results_reshaped_for_analysis[s,t,r,c,d]=predictedResult[varIndex,cellIndex];
                    results_reshaped_for_analysis[s,t+nTrans,r,c,d]=predictedResult[varIndex+nTrans*nStim,cellIndex];
                    results_reshaped_for_analysis[s,t+nTrans*2,r,c,d]=predictedResult[varIndex+(nTrans*nStim)*2,cellIndex];
                    cellIndex +=1; 
        varIndex += 1;


  
IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis,results_reshaped_for_analysis,plotOn=False,nBins=3)
# IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True,thresholdMode=True)
# IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis_avg(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True)
plotting.plotActivityOfCellsWithMaxInfo(IRs=IRs_weighted_list[1],results=results_reshaped_for_analysis);


results_all_trained = results_reshaped_for_analysis;









### info analysis (video only)
# info analysis for all V only

#create a combined input to test combinations of A and V (V only, A only, and V+A)
xTrain_comb = np.zeros((nTrainSet*3,dim*dim*2));
xTrain_comb[:nTrainSet,:dim*dim] = XVisual
xTrain_comb[:nTrainSet,dim*dim:dim*dim*2] = emptyInput



predictedResult = pca.transform(xTrain_comb);
results_reshaped_for_analysis = np.zeros((nStim,nTrans,nRow,nCol,nDep));
varIndex = 0;
for s in range(nStim):
    for t in range(nTrans):
        cellIndex=0;
        for r in range(nRow):
            for c in range(nCol):
                for d in range(nDep):
                    results_reshaped_for_analysis[s,t,r,c,d]=predictedResult[varIndex,cellIndex];
                    cellIndex +=1; 
        varIndex += 1;
    

IRs_list_VOnly, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis,results_reshaped_for_analysis,plotOn=True,nBins=3)
# IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True,thresholdMode=True)
# IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis_avg(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True)
# plotting.plotActivityOfCellsWithMaxInfo(IRs=IRs_weighted_list[1],results=results_reshaped_for_analysis_trained,title="V Only");
plotting.plotActivityOfCellsWithMaxInfo(IRs=IRs_weighted_list[1],results=results_all_trained,title="based on info about V");






### info analysis (audio only)
# info analysis for all A only

#create a combined input to test combinations of A and V (V only, A only, and V+A)
xTrain_comb = np.zeros((nTrainSet*3,dim*dim*2));
xTrain_comb[:nTrainSet,:dim*dim] = emptyInput 
xTrain_comb[:nTrainSet,dim*dim:dim*dim*2] = XAudio



predictedResult = pca.transform(xTrain_comb);
results_reshaped_for_analysis = np.zeros((nStim,nTrans,nRow,nCol,nDep));
varIndex = 0;
for s in range(nStim):
    for t in range(nTrans):
        cellIndex=0;
        for r in range(nRow):
            for c in range(nCol):
                for d in range(nDep):
                    results_reshaped_for_analysis[s,t,r,c,d]=predictedResult[varIndex,cellIndex];
                    cellIndex +=1; 
        varIndex += 1;
    

IRs_list_AOnly, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis,results_reshaped_for_analysis,plotOn=True,nBins=3)
# IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True,thresholdMode=True)
# IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis_avg(results_reshaped_for_analysis_untrained,results_reshaped_for_analysis_trained,plotOn=True)
# plotting.plotActivityOfCellsWithMaxInfo(IRs=IRs_weighted_list[1],results=results_reshaped_for_analysis_trained,title="V Only");
plotting.plotActivityOfCellsWithMaxInfo(IRs=IRs_weighted_list[1],results=results_all_trained,title="based on info about A");
















# IRs_list, IRs_weighted_list = analysis.singleCellInfoAnalysis(results,results,plotOn=True,thresholdMode = False, nBins=3,threshold = 0.7)








# # inverse
# Xe = pca.inverse_transform(predictedResult)
# print(Xe.shape)
# 
# 
# fig, axes = plt.subplots(ncols=10, nrows=4, figsize=(30,4))
# 
# 
# for i in range(10):
#     #plot V
#     axes[0, i].imshow(X[i*50,:28*28].reshape(28,28),vmin=0.0,vmax=1.0, cmap = cm.Greys_r)
#     axes[0, i].set_title('original %d' % i)
#     axes[0, i].get_xaxis().set_visible(False)
#     axes[0, i].get_yaxis().set_visible(False)
# 
#     axes[1, i].imshow(Xe[i*50,:28*28].reshape(28,28),vmin=0.0,vmax=1.0, cmap = cm.Greys_r)
#     axes[1, i].set_title('dimension reduction %d' % i)
#     axes[1, i].get_xaxis().set_visible(False)
#     axes[1, i].get_yaxis().set_visible(False)
#     
#     #plot A
#     axes[2, i].imshow(X[i*50,28*28:28*28*2].reshape(28,28),cmap = cm.Greys_r)
#     axes[2, i].set_title('original %d' % i)
#     axes[2, i].get_xaxis().set_visible(False)
#     axes[2, i].get_yaxis().set_visible(False)
# 
#     axes[3, i].imshow(Xe[i*50,28*28:28*28*2].reshape(28,28),cmap = cm.Greys_r)
#     axes[3, i].set_title('dimension reduction %d' % i)
#     axes[3, i].get_xaxis().set_visible(False)
#     axes[3, i].get_yaxis().set_visible(False)
    
# plt.show()