import pickle
import numpy as np
import pylab as plt

nObj = 10;

# for l in [3]:# range(4):
for l in range(4):

#     pkl_file = open('data/singleCellInfo_inconsistent_oneModalityAtTime_bin10_l'+str(l+1)+'.pkl', 'rb')
    pkl_file = open('data/singleCellInfo_consistent_shuffled_bin10_l'+str(l+1)+'.pkl', 'rb')
    data_shuffled = pickle.load(pkl_file)
    pkl_file.close();
    
    #    pkl_file = open('data/mutualInfo_bin50_l'+str(i+1)+'.pkl', 'rb')
    pkl_file = open('data/singleCellInfo_consistent_bin10_l'+str(l+1)+'.pkl', 'rb')
    data_consistent = pickle.load(pkl_file)
    pkl_file.close();
    
#     pkl_file = open('data/singleCellInfo_consistent_oneModalityAtTime_bin10_l'+str(l+1)+'.pkl', 'rb')
#     data_oneModalityAtTime = pickle.load(pkl_file)
#     pkl_file.close();
    
    IRs = data_shuffled[0]
    IRs_flattened_shuffled = np.zeros((nObj,64));
    for obj in range(nObj):
        IRs_flattened_shuffled[obj] = IRs[obj].flatten();    
    IRs_sorted_shuffled = np.sort(IRs_flattened_shuffled*-1)*-1;
    
    IRs = data_consistent[0]
    IRs_flattened_untrained = np.zeros((nObj,64));
    for obj in range(nObj):
        IRs_flattened_untrained[obj] = IRs[obj].flatten();    
    IRs_sorted_untrained = np.sort(IRs_flattened_untrained*-1)*-1;
    
    
    IRs = data_consistent[1]
    IRs_flattened_trained = np.zeros((nObj,64));
    for obj in range(nObj):
        IRs_flattened_trained[obj] = IRs[obj].flatten();    
    IRs_sorted_trained = np.sort(IRs_flattened_trained*-1)*-1;
       
    # plt.subplot(4,1,1)
    # plt.plot(np.transpose(IRs_sorted_untrained));
    # plt.ylabel("single cell info [bit]")
    # plt.xlabel("cell rank")
    # plt.title("untrained network")
    # plt.ylim((np.log2(nObj)*-0.05,np.log2(nObj)*1.05))
    # 
    # plt.subplot(4,1,2)
    # plt.plot(np.transpose(IRs_sorted_trained_inconsistent)); 
    # plt.ylabel("single cell info [bit]")
    # plt.xlabel("cell rank")
    # plt.title("trained network (inconsistent)")
    # plt.ylim((np.log2(nObj)*-0.05,np.log2(nObj)*1.05))
    # plt.subplot(4,1,3)   
    # plt.plot(np.transpose(IRs_sorted_trained_consistent)); 
    # plt.ylabel("single cell info [bit]")
    # plt.xlabel("cell rank")
    # plt.title("trained network (consistent)")
    # plt.ylim((np.log2(nObj)*-0.05,np.log2(nObj)*1.05))
    
    plt.subplot(4,1,4-l)   
    plt.plot(np.transpose(np.sort(np.max(IRs_flattened_shuffled,axis=0)*-1)*-1),':',label='Shuffled')
    plt.plot(np.transpose(np.sort(np.max(IRs_flattened_untrained,axis=0)*-1)*-1),'--',label='Untrained')
    plt.plot(np.transpose(np.sort(np.max(IRs_flattened_trained,axis=0)*-1)*-1),label='Trained (Epoch 5000)')
    plt.ylabel("single cell info [bit]")
    plt.xlabel("cell rank")
#     plt.title("Layer " + str(l+1))
    plt.title("Single Cell Information Analysis")
    plt.ylim((np.log2(nObj)*-0.05,np.log2(nObj)*1.05))

# plt.subplot(2,1,2)
# plt.plot(np.sort(np.max(IRs_flattened_untrained,axis=0)*-1)*-1)
plt.legend()
plt.subplots_adjust(hspace=1.0)
plt.show()