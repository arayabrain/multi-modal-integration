import pickle
import numpy as np
import pylab as plt




trained_consistent = []
trained_inconsistent = []
untrained = []
trained_consistent_shuffled = []
for i in range(4):

#    pkl_file = open('data/mutualInfo_bin50_l'+str(i+1)+'.pkl', 'rb')
    pkl_file = open('data/singleCellInfo_consistent_bin10_l'+str(i+1)+'.pkl', 'rb')
#     pkl_file = open('data/singleCellInfo_consistent_oneModalityAtTime_bin10_l'+str(i+1)+'.pkl', 'rb')
    data_consistent = pickle.load(pkl_file)
    pkl_file.close();

    pkl_file = open('data/singleCellInfo_inconsistent_bin10_l'+str(i+1)+'.pkl', 'rb')
#     pkl_file = open('data/singleCellInfo_inconsistent_oneModalityAtTime_bin10_l'+str(i+1)+'.pkl', 'rb')
    data_inconsistent = pickle.load(pkl_file)
    pkl_file.close();

#     pkl_file = open('data/singleCellInfo_consistent_oneModalityAtTime_shuffled_bin10_l'+str(i+1)+'.pkl', 'rb')
    pkl_file = open('data/singleCellInfo_consistent_shuffled_bin10_l'+str(i+1)+'.pkl', 'rb')
    data_consistent_shuffled = pickle.load(pkl_file)
    pkl_file.close();


    
#     SI_untrained = np.sort(np.max(data_consistent[0],axis=0)*-1)*-1
#     SI_trained = np.sort(np.max(data_consistent[1],axis=0)*-1)*-1;  
    
    IRs_flattened_untrained = np.zeros((10,64));
    IRs_flattened_trained_consistent = np.zeros((10,64));
    IRs_flattened_trained_inconsistent = np.zeros((10,64));
    IRs_flattened_trained_consistent_shuffled = np.zeros((10,64));

    for obj in range(10):
        IRs_flattened_untrained[obj] = data_consistent[0][obj].flatten();     
        IRs_flattened_trained_consistent[obj] = data_consistent[1][obj].flatten();
        IRs_flattened_trained_inconsistent[obj] = data_inconsistent[1][obj].flatten();
        IRs_flattened_trained_consistent_shuffled[obj] = data_consistent_shuffled[0][obj].flatten();
        
    untrained.append(np.mean(np.max(IRs_flattened_untrained,axis=0)))
    trained_consistent.append(np.mean(np.max(IRs_flattened_trained_consistent,axis=0)))
    trained_inconsistent.append(np.mean(np.max(IRs_flattened_trained_inconsistent,axis=0)))
    trained_consistent_shuffled.append(np.mean(np.max(IRs_flattened_trained_consistent_shuffled,axis=0)))
      
#     plt.subplot(4,1,4-i)
#     plt.title("Layer "+str(i+1));
#     plt.plot(np.sort(np.max(IRs_flattened_untrained,axis=0)*-1)*-1,label="SI epoch 0 (layer "+str(i+1)+")");    
#     plt.plot(np.sort(np.max(IRs_flattened_trained_consistent,axis=0)*-1)*-1,label="SI epoch 5000 (layer "+str(i+1)+")")
#     plt.ylabel("Information [bit]")
#     plt.xlabel("Cell rank")
#     plt.ylim([-0.1,np.log2(10)+0.1])
#     plt.legend() 

#     print("Layer " + str(i) + " untrained " + str(np.mean(MA_untrained)) + " trained_consistent " + str(np.mean(MA_trained)));
   
plt.plot(["layer 1","layer 2","layer 3", "layer 4"],trained_consistent_shuffled,':',label="Avg. SI (shuffled)")
plt.plot(untrained,'--',label="Avg. SI (Epoch 0)")   
# plt.plot(trained_inconsistent,'--',label="Avg. SI - inconsistent (Epoch 5000)")
plt.plot(trained_consistent,label="Avg. SI (Epoch 5000)")
plt.ylabel("Information [bit]")
plt.xlabel("Layer")
plt.title("Average single cell information of each layer")
plt.subplots_adjust(hspace=0.8)
plt.legend()
plt.show()

