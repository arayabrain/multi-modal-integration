import pylab as plt
import numpy as np
import pickle

itrList = [0,1000,2000,3000,4000,5000];
lineType = [':','-.','--','-'];


for i in range(4):
    pkl_file = open('data/mutualInfo_bin20_l'+str(i+1)+'_alongEpochs.pkl', 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close();
    
     
    IV = data[0];
    IA = data[1];
    MI = np.mean(data,axis=0);

    plt.plot(itrList,MI,lineType[i],label="Avg. MI (input x layer "+str(i+1)+")");
# plt.plot(itrList,IA)
plt.ylabel("Information [bit]")
plt.xlabel("Epoch")
plt.title("Average mutual information")
plt.legend();
plt.show()    
    
    
    
#     # ## plot the info
#     plt.subplot(4,1,4-i);
#     plt.plot(-np.sort(-MI_untrained.flatten()),':',label="IV_untrained");
#     plt.plot(-np.sort(-MI_shuffled.flatten()),'--',label="IV_shuffled");
#     plt.plot(-np.sort(-MI_trained.flatten()),label="IV_trained");
#     plt.title("Layer "+ str(4-i))
#     plt.xlabel("s-r pair rank");
#     plt.ylabel("Information [bit]")
#     # plt.ylim((max(IV_trained.max(),IV_untrained.max())*-0.1,max(IV_trained.max(),IV_untrained.max())*1.1));
#     plt.ylim([-0.1,1.1])
#     plt.legend();
    
    
#     # ## plot the info (one modality at time)
#     plt.subplot(2,1,1);
#     plt.plot(-np.sort(-IV_trained.flatten()),label="IV_trained");
#     plt.plot(-np.sort(-IV_untrained.flatten()),label="IV_untrained");
#     plt.title("Visual Input Unit x Encoded Unit")
#     plt.xlabel("s-r pair rank");
#     plt.ylabel("Mutual Information [bit]")
#     # plt.ylim((max(IV_trained.max(),IV_untrained.max())*-0.1,max(IV_trained.max(),IV_untrained.max())*1.1));
#     plt.ylim([-0.1,1.1])
#     plt.legend();
#            
#     plt.subplot(2,1,2);
#     plt.plot(-np.sort(-IA_trained.flatten()),label="IA_trained");
#     plt.plot(-np.sort(-IA_untrained.flatten()),label="IA_untrained");
#     plt.title("Audio Input Unit x Encoded Unit")
#     plt.xlabel("s-r pair rank");
#     plt.ylabel("Mutual Information [bit]")
#     # plt.ylim((max(IA_trained.max(),IA_untrained.max())*-0.1,max(IA_trained.max(),IA_untrained.max())*1.1));
#     plt.ylim([-0.1,1.1])
    
    
# plt.legend();
# plt.subplots_adjust(hspace=1.0)
# plt.show()