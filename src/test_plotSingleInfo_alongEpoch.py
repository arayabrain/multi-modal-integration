import pickle
import numpy as np
import pylab as plt




#    pkl_file = open('data/mutualInfo_bin50_l'+str(i+1)+'.pkl', 'rb')
itrList = [0,1000,2000,3000,4000,5000];
lineType = [':','-.','--','-'];

for l in range(4):

    pkl_file = open('data/SingleCellInfo_bin10_l'+str(l+1)+'_alongEpochs.pkl', 'rb')
#     pkl_file = open('data/singleCellInfo_consistent_oneModalityAtTime_bin10_l'+str(l+1)+'_alongEpochs.pkl', 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close();
    
    plt.plot(itrList,data,lineType[l],label="Avg. SI (Layer "+str(l+1)+")")   
    plt.ylabel("Information [bit]")
    plt.xlabel("Epoch")
    plt.title("Average single cell information at each epoch")

plt.subplots_adjust(hspace=0.8)
plt.legend()
plt.show()

