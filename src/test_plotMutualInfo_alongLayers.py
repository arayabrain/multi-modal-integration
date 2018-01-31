import pylab as plt
import numpy as np
import pickle

layerList =['Layer 1','Layer 2','Layer 3','Layer 4'];

MI_list_untrained = [];
MI_list_trained = [];
MI_list_shuffled = [];
MI_list_untrained_top10 = [];
MI_list_trained_top10 = [];
MI_list_shuffled_top10 = [];


for l in range(4):
    pkl_file = open('data/mutualInfo_bin20_l'+str(l+1)+'.pkl', 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close();
     
     
    IV_untrained = data[0];
    IA_untrained = data[1];    
    IV_trained = data[2];
    IA_trained = data[3];
    
    MI_untrained = np.concatenate((IV_untrained.flatten(),IA_untrained.flatten()))
    MI_trained = np.concatenate((IV_trained.flatten(),IA_trained.flatten()))
    nPairs=np.shape(MI_untrained)[0];
    
    MI_list_untrained.append(np.mean(MI_untrained))
    MI_list_trained.append(np.mean(MI_trained))
    MI_list_untrained_top10.append(np.mean(np.sort(MI_untrained)[-int(nPairs*0.1):]))
    MI_list_trained_top10.append(np.mean(np.sort(MI_trained)[-int(nPairs*0.1):]))
    
    
    
    pkl_file = open('data/mutualInfo_bin20_shuffled_l'+str(l+1)+'.pkl', 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close();
     
    IV_shuffled = data[0];
    IA_shuffled = data[1];    
    
    MI_shuffled = np.concatenate((IV_shuffled.flatten(),IA_shuffled.flatten()))
    MI_list_shuffled.append(np.mean(MI_shuffled))
    MI_list_shuffled_top10.append(np.mean(np.sort(MI_shuffled)[-int(nPairs*0.1):]))
    
    
     
plt.subplot(1,2,1);
plt.plot(layerList,MI_list_untrained_top10,":",label="Avg. MI (untrained)");
plt.plot(layerList,MI_list_shuffled_top10,"--",label="Avg. MI (shuffled)");
plt.plot(layerList,MI_list_trained_top10,label="Avg. MI (trained)");

# plt.plot(itrList,IA)
plt.ylabel("Information [bit]")
plt.xlabel("Layer")
plt.title("Avg. mutual information over top 10 % of the pairs")
plt.legend();

plt.subplot(1,2,2);
plt.plot(layerList,MI_list_untrained,":",label="Avg. MI (untrained)");
plt.plot(layerList,MI_list_shuffled,"--",label="Avg. MI (shuffled)");
plt.plot(layerList,MI_list_trained,label="Avg. MI (trained)");
# plt.plot(itrList,IA)
plt.ylabel("Information [bit]")
plt.xlabel("Layer")
plt.title("Avg. mutual information over all pairs")
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